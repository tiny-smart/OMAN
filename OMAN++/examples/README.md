# Video Demo

Place your video clips (`.mp4`, `.avi`, `.mov`, `.mkv`, etc.) here, activate your
VIC environment, and run from the repository root:

```bash
python demo.py
```

Or pass a video file directly:

```bash
python demo.py --input /path/to/clip.mp4
```

The model defaults to `pretrained/flux.pth`. No annotations or intermediate frame
extraction are required. Videos are read incrementally, not loaded into memory
all at once. Input videos are not bundled.

## Outputs

The default output directory is `examples/results`:

- `<input_filename>_result.mp4`: annotated video with head points, frame counts,
  cumulative counts, and newly observed counts. Audio is not copied.
- `results.json`: results for each video, with sampled frame indices (zero-based),
  source timestamps, original-image `[x, y]` coordinates, frame counts, inflow
  flags, cumulative counts, and inference times.

Each video has independent temporal memory. Subdirectories are scanned
recursively, their relative structure is preserved, and the output directory is
excluded from input discovery. Red points are newly observed people; green
points are previously observed people. Cumulative counts use historical feature
matching, not directional line crossings or persistent track IDs.

## Options

```bash
python demo.py --input clip.mp4 --device cuda:0 --interval 5
python demo.py --input clip.mp4 --start-time 10 --end-time 20
python demo.py --input clip.mp4 --device cpu --size 384 512 --max-frames 2
python demo.py --input clip.mp4 --localization-only --output ./video_results
python demo.py --help
```

- `--interval N` infers and saves only every Nth frame (default: 50). Intervening
  frames are not saved. Output FPS is source FPS / N, keeping the sampled frames
  at their original time spacing rather than speeding up playback. JSON has one
  entry per saved frame; `fps` is source FPS and `output_fps` is output FPS.
- `--start-time` and `--end-time` select a time range in seconds, relative to the
  original video. The end is exclusive; the first frame timestamp is at or after
  the start. Output playback starts at zero, while JSON keeps source timestamps.
- `--max-frames N` stops the clip immediately after the Nth inference, so the
  output may be shorter than the requested video range.
- `--size HEIGHT WIDTH` defaults to `768 1024`; both must be positive multiples
  of 64. Coordinates map back to original video dimensions. Odd video dimensions
  receive one black padding row/column for MP4 encoding.
- `--device auto` chooses CUDA when available, otherwise CPU. CPU is slower.
- `--localization-only` skips temporal matching and exports per-frame counts.
- `--checkpoint PATH` selects a trusted, compatible local checkpoint. The demo
  adapts legacy attention weights and does not load extra backbone weights.
  Only load trusted checkpoints because deserialization can execute code.

Timing excludes video decoding, preprocessing, annotation, and encoding. Memory
and matching costs grow with the number of people retained in temporal memory;
use short clips for this quick demo. Counts can change with inference resolution
and sampling interval.
