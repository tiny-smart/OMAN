import random
import json
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset
from models import build_model
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='convnext', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)  # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)  # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")  # cross-entropy weights

    # dataset parameters
    parser.add_argument('--dataset_file', default="SENSE")
    parser.add_argument('--test_root', default='/data/SENSE/test')
    parser.add_argument('--ann_dir', default='/data/SENSE/label_list_all')
    parser.add_argument('--max_len', default=3000)

    # misc parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu', default='0,1,2,3', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--resume', default='./outputs/SENSE/exp_VIC/checkpoint1.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='pretrained/SENSE.pth', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="./outputs/SENSE/img_VIC")
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def read_pts(model, img):
    if isinstance(img, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(img.unsqueeze(0).cuda())
    points, features = model(samples, [], [], test=True)
    return points, features['4x'].tensors

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # initilize the model
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        sync_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            sync_model, device_ids=[args.gpu], find_unused_parameters=True)  # default: False
        model_without_ddp = model.module

    # build dataset
    sharing_strategy = "file_system"
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    def set_worker_sharing_strategy(worker_id: int) -> None:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    dataset_test = build_dataset(args.dataset_file, args.test_root, args.ann_dir)  # default step = 15

    sampler_val = DistributedSampler(dataset_test, shuffle=False) if args.distributed else None

    data_loader_val = DataLoader(dataset_test,
                                 batch_size=1,
                                 sampler=sampler_val,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 worker_init_fn=set_worker_sharing_strategy)

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    model.eval()
    video_results = {}
    interval = 15

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader_val):
            cnt_list = []
            video_name = labels["video_name"][0]
            img_names = labels["img_names"]
            w, h = labels["w"][0], labels["h"][0]

            img_name0 = img_names[0][0]
            pos_path0 = os.path.join(
                "locator", video_name, img_name0 + ".txt")
            print(pos_path0)

            pos0, feature0 = read_pts(model, imgs[0, 0])

            if args.distributed:
                z0 = model.module.forward_single_image(
                    imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
            else:
                z0 = model.forward_single_image(
                    imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
            pre_z = z0
            pre_pos = pos0
            cnt_0 = len(pos0)
            cum_cnt = cnt_0
            cnt_list.append(cnt_0)
            selected_idx = [v for v in range(
                interval, len(img_names), interval)]
            pos_lists = []
            inflow_lists = []
            outflow_lists = []
            pos_lists.append(pos0)
            inflow_lists.append([1 for _ in range(len(pos0))])
            for i in selected_idx:
                pos, feature1 = read_pts(model, imgs[0, i])

                pre_pre_z = pre_z
                if args.distributed:
                    z1, z2, pre_z = model.module.forward_single_image(
                        imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z)
                else:
                    z1, z2, pre_z = model.forward_single_image(
                        imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z)
                z1 = F.normalize(z1, dim=-1).transpose(0, 1)
                z2 = F.normalize(z2, dim=-1).transpose(0, 1)

                ''' einsum '''
                sim_feats = torch.einsum('bnc,bmc->bnmc', z2, z1)  # [1, n, m, c]
                sim_feats = sim_feats.view(1, -1, z1.shape[-1])  # [1, n*m, c]

                if args.distributed:
                    pred_logits = model.module.vic.regression(sim_feats.squeeze(0))  # [n*m, num_classes]
                else:
                    pred_logits = model.vic.regression(sim_feats.squeeze(0))  # [n*m, num_classes]
                pred_probs = F.softmax(pred_logits, dim=1)  # [n*m, num_classes]
                pred_scores, pred_classes = pred_probs.max(dim=1)  # [n*m]

                pedestrian_idx = torch.nonzero(pred_classes == 0).squeeze(1).cpu().numpy()

                pedestrian_list = pedestrian_idx // z1.shape[1]
                pre_pedestrian_list = pedestrian_idx % z1.shape[1]

                inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
                outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]


                pos_lists.append(pos)
                inflow_list = []
                for j in range(len(pos)):
                    if j in inflow_idx_list:
                        inflow_list.append(1)
                    else:
                        inflow_list.append(0)
                inflow_lists.append(inflow_list)
                cum_cnt += len(inflow_idx_list)
                cnt_list.append(len(inflow_idx_list))

                outflow_list = []
                for j in range(len(pre_pos)):
                    if j in outflow_idx_list:
                        outflow_list.append(1)
                    else:
                        outflow_list.append(0)
                outflow_lists.append(outflow_list)

                z_mask = np.array(outflow_list, dtype = bool)
                mem = pre_pre_z[0][:len(pre_pos)][z_mask]
                pre_z = [torch.cat((pre_z[0], mem), dim=0)]
                pre_pos = pos

            # conver numpy to list
            pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]

            video_results[video_name] = {
                "video_num": cum_cnt,
                "first_frame_num": cnt_0,
                "cnt_list": cnt_list,
                "frame_num": len(img_names),
                "pos_lists": pos_lists,
                "inflow_lists": inflow_lists,

            }
            print(video_name, video_results[video_name]["video_num"],video_results[video_name]["cnt_list"])

    with open("outputs/json/video_results_test.json", "w") as f:
        json.dump(video_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
