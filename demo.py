"""Run VIC crowd counting on video clips and export annotated videos and JSON."""
import argparse
import json
import math
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv', '.ts'}


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/test.mp4',
                        help='A video file or directory of videos (default: ./examples)')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'pretrained/WuhanMetroCrowd.pth',
                        help='Trusted local VIC checkpoint')
    parser.add_argument('--output', type=Path, default=ROOT / 'examples/results')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--size', nargs=2, type=int, default=[768, 1024],
                        metavar=('HEIGHT', 'WIDTH'), help='Inference resolution (default: 768 1024)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Point confidence threshold in [0, 1]')
    parser.add_argument('--interval', type=int, default=50, help='Infer and save every Nth video frame')
    parser.add_argument('--max-frames', type=int, help='Stop each clip after N inferred frames')
    parser.add_argument('--start-time', type=float, default=0, help='Start at this time in seconds')
    parser.add_argument('--end-time', type=float, help='Stop before this time in seconds')
    parser.add_argument('--localization-only', action='store_true',
                        help='Skip temporal matching; only predict per-frame points and counts')
    return parser


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(path))]


def collect_videos(input_path, output_path):
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError('Input must be a video file or a directory of videos.')
        return {input_path.name: input_path}
    if not input_path.is_dir():
        raise FileNotFoundError('Input directory does not exist: {}'.format(input_path))
    if input_path == output_path or output_path in input_path.parents:
        raise ValueError('The output directory must not contain the input directory.')
    videos = {}
    for path in sorted(input_path.rglob('*'), key=natural_key):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if output_path == path.parent or output_path in path.parents:
            continue
        videos[path.relative_to(input_path).as_posix()] = path
    if not videos:
        raise ValueError('No videos found in {}. Add a video clip or use --input VIDEO_PATH.'.format(input_path))
    return videos


def load_model(checkpoint_path, device):
    import torch
    from models.backbones import build_backbone_vgg
    from models.pet import PET
    from models.transformer.transformer import MultiHeadLatentAttention

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint.get('args', {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    defaults = dict(backbone='convnext-s', position_embedding='sine', dec_layers=2,
                    dim_feedforward=512, hidden_dim=256, dropout=0.0, nheads=8)
    defaults.update({name: saved_args[name] for name in defaults if name in saved_args})
    if defaults['backbone'] == 'convnext':
        defaults['backbone'] = 'convnext-s'
    args = argparse.Namespace(**defaults, pretrained_backbone=False)
    model = PET(build_backbone_vgg(args), num_classes=1, args=args)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {name.removeprefix('module.'): value for name, value in state.items()}
    for index, layer in enumerate(model.vic.encoder.layers):
        prefix = 'vic.encoder.layers.{}.self_attn.'.format(index)
        if prefix + 'Wq_d.weight' in state:
            query_dim, model_dim = state[prefix + 'Wq_d.weight'].shape
            latent_dim = state[prefix + 'Wkv_d.weight'].shape[0]
            heads = state[prefix + 'W_qk.weight'].shape[0] // latent_dim
            layer.self_attn = MultiHeadLatentAttention(
                model_dim, heads, q_latent_dim=query_dim, kv_latent_dim=latent_dim,
                dropout=args.dropout)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess(image, size, device):
    import numpy as np
    import torch
    from PIL import Image

    resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def update_flow(model, image_tensor, points, features, memory, memory_points):
    import numpy as np
    import torch
    from torch.nn import functional as functional
    from models.vic import compute_relative_position

    if len(points) == 0:
        return [], memory, memory_points
    if memory is None:
        memory = model.forward_single_image(image_tensor, [points], features, True)
        return [True] * len(points), memory, points.copy()
    previous, current, current_memory, _ = model.forward_single_image(
        image_tensor, [points], features, True, memory, [memory_points])
    previous = functional.normalize(previous, dim=-1).transpose(0, 1)
    current = functional.normalize(current, dim=-1).transpose(0, 1)
    similarity = torch.einsum('bnc,bmc->bnmc', current, previous).flatten(1, 2)
    height, width = image_tensor.shape[-2:]
    current_positions = torch.as_tensor(points[:, [1, 0]] / [height, width], dtype=torch.float32)
    previous_positions = torch.as_tensor(memory_points[:, [1, 0]] / [height, width], dtype=torch.float32)
    _, difference = compute_relative_position(current_positions, previous_positions)
    distance = model.vic.dist_encoder(difference.to(image_tensor.device).transpose(1, 2)).transpose(1, 2)
    distance = functional.normalize(distance, dim=2).reshape(1, -1, 256)
    classes = model.vic.regression(similarity.squeeze(0), distance).argmax(dim=1)
    matched = torch.nonzero(classes == 0, as_tuple=False).flatten().cpu().numpy()
    incoming = ~np.isin(np.arange(len(points)), matched // len(memory_points))
    unmatched = ~np.isin(np.arange(len(memory_points)), matched % len(memory_points))
    keep = torch.as_tensor(unmatched, device=image_tensor.device)
    memory = [torch.cat((current_memory[0], memory[0][keep]), dim=0)]
    memory_points = np.concatenate((points, memory_points[unmatched]), axis=0)
    return incoming.tolist(), memory, memory_points


def draw_result(image, points, incoming, count, cumulative):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    radius = max(2, round(min(image.size) / 250))
    for index, (point_x, point_y) in enumerate(points):
        color = 'red' if incoming is not None and incoming[index] else 'lime'
        draw.ellipse((point_x - radius, point_y - radius, point_x + radius, point_y + radius), fill=color)
    text = 'Count: {}'.format(count)
    if cumulative is not None:
        text += ' | Cumulative: {} | New: {}'.format(cumulative, sum(incoming))
    bounds = draw.textbbox((8, 8), text)
    draw.rectangle((0, 0, bounds[2] + 8, bounds[3] + 8), fill='black')
    draw.text((8, 8), text, fill='white')
    return canvas


def infer_video(model, path, destination, args, device):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from util.misc import nested_tensor_from_tensor_list

    capture = cv2.VideoCapture(str(path))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open video: {}'.format(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has invalid FPS: {}'.format(path))
        output_fps = fps / args.interval
        first_frame = math.ceil(args.start_time * fps)
        end_frame = None if args.end_time is None else math.ceil(args.end_time * fps)
        if first_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame):
            raise RuntimeError('Cannot seek to --start-time in {}'.format(path))
        frame_index = first_frame
        memory, memory_points = None, None
        cumulative = 0
        frames = []
        written = 0
        original_points = np.empty((0, 2), dtype=np.float32)
        incoming = None
        original_size = None
        while end_frame is None or frame_index < end_frame:
            success, pixels = capture.read()
            if not success:
                break
            height, width = pixels.shape[:2]
            if writer is None:
                original_size = [width, height]
                output_size = (width + width % 2, height + height % 2)
                destination.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, output_size)
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output video: {}'.format(destination))
            elif original_size != [width, height]:
                raise ValueError('Video frame dimensions changed in {}'.format(path))
            if (frame_index - first_frame) % args.interval == 0:
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image_tensor = preprocess(image, args.size, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                predictions, features = model(nested_tensor_from_tensor_list(image_tensor), [], [],
                                              test=True, threshold=args.threshold)
                normalized = predictions['pred_points'][0].cpu().numpy()
                valid = np.isfinite(normalized).all(axis=1) & (normalized >= 0).all(axis=1) & (normalized < 1).all(axis=1)
                points = (normalized[valid][:, [1, 0]] * [args.size[1], args.size[0]]).astype(np.float32)
                if not args.localization_only:
                    incoming, memory, memory_points = update_flow(
                        model, image_tensor, points, features, memory, memory_points)
                    cumulative += sum(incoming)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                original_points = points * [width / args.size[1], height / args.size[0]]
                frame = dict(frame_index=frame_index, timestamp_seconds=frame_index / fps,
                             count=len(points), points=original_points.tolist(),
                             inference_seconds=round(elapsed, 4))
                if incoming is not None:
                    frame.update(inflow=sum(incoming), inflow_flags=incoming, cumulative_count=cumulative)
                frames.append(frame)
                print('[{}] frame {} ({:.3f}s): {} people, {:.3f}s{}'.format(
                    path.name, frame_index, frame_index / fps, len(points), elapsed,
                    '' if args.localization_only else ', cumulative={}'.format(cumulative)), flush=True)
                canvas = draw_result(image, original_points, incoming, len(original_points),
                                     None if args.localization_only else cumulative)
                annotated = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
                annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(annotated)
                written += 1
            frame_index += 1
            if args.max_frames is not None and len(frames) >= args.max_frames:
                break
        if written == 0:
            raise ValueError('No readable frames in the requested time range: {}'.format(path))
        result = dict(video=str(path), visualization=str(destination), fps=fps, output_fps=output_fps,
                      original_size=original_size, output_size=list(output_size),
                      start_frame=first_frame, end_frame_exclusive=frame_index,
                      written_frames=written, inferred_frames=len(frames), frames=frames)
        if not args.localization_only:
            result['cumulative_count'] = cumulative
        return result
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if any(value <= 0 or value % 64 for value in args.size):
        parser.error('--size HEIGHT WIDTH must be positive multiples of 64')
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')
    if args.interval <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error('--interval and --max-frames must be positive')
    if not math.isfinite(args.start_time) or args.start_time < 0:
        parser.error('--start-time must be a finite nonnegative number')
    if args.end_time is not None and (not math.isfinite(args.end_time) or args.end_time <= args.start_time):
        parser.error('--end-time must be finite and greater than --start-time')
    try:
        videos = collect_videos(args.input, args.output)
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError('Checkpoint not found: {}'.format(args.checkpoint))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import torch

    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(device_name)
    if device.type not in ('cpu', 'cuda'):
        parser.error('--device must be auto, cpu, cuda, or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable; use --device cpu')
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.backends.cudnn.benchmark = device.type == 'cuda'
    print('Loading {} on {} ...'.format(args.checkpoint, device), flush=True)
    model = load_model(args.checkpoint, device)
    output_path = args.output.expanduser().resolve()
    results = {'checkpoint': str(args.checkpoint), 'size': args.size,
               'threshold': args.threshold, 'localization_only': args.localization_only,
               'interval': args.interval, 'start_time': args.start_time,
               'end_time': args.end_time, 'videos': {}}
    with torch.inference_mode():
        for name, path in videos.items():
            destination = output_path / Path(name).parent / (path.name + '_result.mp4')
            try:
                results['videos'][name] = infer_video(model, path, destination, args, device)
            except (OSError, ValueError, RuntimeError) as error:
                parser.error(str(error))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / 'results.json').open('w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    main()
