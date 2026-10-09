import argparse
import random
import json
import time
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import SyncBatchNorm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from models.vic import compute_relative_position, pos2posemb2d
from my_plot_Metro import draw, draw_pair, draw_failure
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
    # parser.add_argument('--dataset_file', default="SENSE")
    # parser.add_argument('--test_root', default='/data/bjwang/SENSE/test')
    # parser.add_argument('--ann_dir', default='/data/bjwang/SENSE/label_list_all')
    parser.add_argument('--dataset_file', default="Metro")
    parser.add_argument('--test_root', default='/data/bjwang/WuhanMetro/test')
    parser.add_argument('--ann_dir', default='/data/bjwang/WuhanMetro/json')
    parser.add_argument('--max_len', default=3000)

    # misc parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu', default='0,1,2,3', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='outputs/Metro/exp_VIC/checkpoint10.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='outputs/Metro/pretrained/WuhanMetro.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/home/bjwang/TOPO-PET-Trans/pretrained/SENSE_GOAT.pth', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def read_pts(model, img):
    if isinstance(img, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(img.unsqueeze(0).cuda())
    outputs, features = model(samples, [], [], test=True)

    outputs_points = outputs['pred_points'][0].cpu()
    outputs_points = outputs_points.detach().numpy()
    img_h, img_w = samples.tensors.shape[-2:]
    points = [[float(point[1] * img_w), float(point[0] * img_h)] for point in outputs_points]

    if len(points) == 0:
        points.append([1, 1])
    return np.array(points, dtype ='float32'), features['4x'].tensors

def read_pts_from_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            line = line.strip().split(",")
            pts.append([float(line[0]), float(line[1])])
        pts = np.array(pts, dtype='float32')
    return pts

def point_nms(outputs_points, frame):
    # remove nearby points with similar feature
    del_list = []
    del_pts = []
    for row1 in range(len(outputs_points)):
        if row1 not in del_list:
            pts1 = np.array([outputs_points[row1][0], outputs_points[row1][1]])
            for row2 in range(row1 + 1, len(outputs_points)):
                pts2 = np.array([outputs_points[row2][0], outputs_points[row2][1]])
                # 计算相似度来作为阈值条件
                dist = np.linalg.norm(pts1 - pts2)
                if 16 > dist > 0:
                    # f1 = My_Static.last_features_4x[0,:,int(pts1[1]),int(pts1[0])]
                    # f2 = My_Static.last_features_4x[0,:,int(pts2[1]),int(pts2[0])]
                    f1 = frame[:, int(pts2[0]) - int(dist):int(pts2[0] + int(dist)),
                         int(pts2[1]) - int(dist):int(pts2[1]) + int(dist)]
                    f2 = frame[:, int(pts2[0]) - int(dist):int(pts2[0] + int(dist)),
                         int(pts2[1]) - int(dist):int(pts2[1]) + int(dist)]
                    similarity = F.cosine_similarity(f1, f2, dim=0).cpu()
                    if torch.mean(similarity) > 0.8:
                        del_list.append(row2)
                        del_pts.append(np.array([outputs_points[row2][0], outputs_points[row2][1]]))
    outputs_points = np.delete(outputs_points, del_list, axis=0)
    return outputs_points

def save_pts(video_name, img_name, points_list):
    """ save locator results """
    txt_path = './locator/' + video_name
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    f = open(txt_path + '/' + img_name + '.txt', "w")
    for pts in points_list:
        f.write(str(pts[0]) + ',' + str(pts[1]) + '\n')
    f.close()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    plot_flag = 1
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
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

        # """ joint test """
        # pretraind_dict = torch.load('outputs/SENSE/exp_VIC/best_checkpoint.pth', map_location='cpu')
        # model_dict = model.state_dict()
        # # 只将pretraind_dict中那些在model_dict中的参数，提取出来
        # state_dict = {k: v for k, v in pretraind_dict["model"].items()}
        # # 将提取出来的参数更新到model_dict中，而model_dict有的，而state_dict没有的参数，不会被更新
        # model_dict.update(state_dict)
        # pretraind_dict2 = torch.load('outputs/SENSE/exp_VIC/checkpoint2.pth', map_location='cpu')
        # state_dict2 = {k: v for k, v in pretraind_dict2["model"].items() if 'vic' in k and 'mlp' in k}
        # model_dict.update(state_dict2)
        # model_without_ddp.load_state_dict(model_dict)


    model.eval()
    video_results = {}

    if args.dataset_file == 'SENSE':
        interval = 15
    elif args.dataset_file == 'HT21':
        interval = 75
    elif args.dataset_file == 'Metro':
        interval = 1

    start = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader_val):
            cnt_list = []
            video_name = labels["video_name"][0]
            img_names = labels["img_names"]
            # w, h = labels["w"][0], labels["h"][0]
            w, h = 1280, 720
            img_name0 = img_names[0][0]
            pos_path0 = os.path.join(
                "locator", video_name, img_name0 + ".txt")
            print(pos_path0)

            pos0, feature0 = read_pts(model, imgs[0, 0])
            # pos0 = point_nms(pos0, imgs[0, 0])

            # pos0, feature0 = read_pts_from_txt(pos_path0), []   # read points from txt
            # pos0 = point_nms(pos0, imgs[0, 0])    # read points from txt

            # pos0 = np.multiply(labels["pts"][0][0].numpy(),np.array([w, h])).astype(np.float32)   # GT
            # pos0 = point_nms(pos0, imgs[0, 0])   # GT
            # feature0 =[]    # GT

            # save_pts(video_name, img_name0, pos0)
            # feature0 = model.backbone(imgs[0, 0].cuda().unsqueeze(0)) # roi
            if args.distributed:
                z0 = model.module.forward_single_image(
                    imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
            else:
                z0 = model.forward_single_image(
                    imgs[0, 0].cuda().unsqueeze(0), [pos0], feature0, True)
            pre_z = z0
            pre_pos = pos0
            temp = pos0
            pre_mem = torch.tensor([])
            pre_img_name = img_name0
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
            # if selected_idx[-1] != len(img_names)-1:
            #     selected_idx.append(len(img_names)-1)
            for i in selected_idx:
                img_name = img_names[i][0]
                pos_path = os.path.join(
                    "locator", video_name, img_name + ".txt")

                pos, feature1 = read_pts(model, imgs[0, i])
                # pos = point_nms(pos, imgs[0, i])

                # pos, feature1 = read_pts_from_txt(pos_path), []   # read points from txt
                # pos = point_nms(pos, imgs[0, i])

                # pos = np.multiply(labels["pts"][i][0].numpy(),np.array([w, h])).astype(np.float32)     # GT
                # pos = point_nms(pos, imgs[0, i])     # GT
                # feature1 = []  # GT

                # save_pts(video_name, img_name, pos)
                pre_pre_z = pre_z
                # feature1 = model.backbone(imgs[0, i].cuda().unsqueeze(0)) # roi
                if args.distributed:
                    z1, z2, pre_z = model.module.forward_single_image(
                        imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z, [pre_pos])
                else:
                    z1, z2, pre_z, attention_map = model.forward_single_image(
                        imgs[0, i].cuda().unsqueeze(0), [pos], feature1, True, pre_z, [pre_pos])
                z1 = F.normalize(z1, dim=-1).transpose(0, 1)
                z2 = F.normalize(z2, dim=-1).transpose(0, 1)

                # visualize attention map or similarity map
                # from my_demo.visualize_feature import vis_attn
                # vis_attn(imgs[0, i-interval].cuda().unsqueeze(0), imgs[0, i].cuda().unsqueeze(0), attention_map, pre_pos, pos, 3)
                # sim_map = torch.einsum('bnc,bmc->bnm', torch.cat((z1, z2), dim=1), torch.cat((z1, z2), dim=1))
                # vis_attn(imgs[0, i-interval].cuda().unsqueeze(0), imgs[0, i].cuda().unsqueeze(0), sim_map, pre_pos, pos, 1)

                ''' Hungarian '''
                # match_matrix = torch.bmm(z1, z2.transpose(1, 2))
                #
                # C = match_matrix.cpu().detach().numpy()[0]
                # row_ind, col_ind = linear_sum_assignment(-C)

                ''' Hungarian Only, No MLP '''
                # sim_score = C[row_ind, col_ind]
                # shared_mask = sim_score > 0.4
                # pre_pedestrian_list = row_ind[shared_mask]
                # pedestrian_list = col_ind[shared_mask]
                # inflow_idx_list = [i for i in range(len(pos)) if i not in col_ind[shared_mask]]
                # outflow_idx_list = [i for i in range(len(pos)) if i not in row_ind[shared_mask]]

                ''' Hungarian, MLP '''
                # sim_feat = z1[:, row_ind, :] * z2[:, col_ind, :]
                # if args.distributed:
                #     pred_logits = model.module.vic.regression(sim_feat.squeeze(0))
                # else:
                #     pred_logits = model.vic.regression(sim_feat.squeeze(0))
                # pred_prob = F.softmax(pred_logits, dim=1)
                # pred_score, pred_class = pred_prob.max(dim=1)
                # pedestrian_list = col_ind[(1 - pred_class).bool().cpu().numpy()]
                # pre_pedestrian_list = row_ind[(1 - pred_class).bool().cpu().numpy()]
                # inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
                # outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]
                # if inflow_idx_list:
                #     for idx2 in inflow_idx_list:
                #         for idx1 in range(z1.shape[1]):
                #             sim_feat2 = z1[:, idx1, :] * z2[:, idx2, :]
                #             if args.distributed:
                #                 pred_logits2 = model.module.vic.regression(sim_feat2.squeeze(0))
                #             else:
                #                 pred_logits2 = model.vic.regression(sim_feat2.squeeze(0))
                #             pred_prob2 = F.softmax(pred_logits2, dim=0)
                #             pred_score2, pred_class2 = pred_prob2.max(dim=0)
                #             if pred_class2 == 0:
                #                 pedestrian_list = np.append(pedestrian_list, idx2)
                #                 pre_pedestrian_list = np.append(pre_pedestrian_list, idx1)
                #                 break
                # inflow_idx_list = [i for i in range(len(pos)) if i not in pedestrian_list]
                # outflow_idx_list = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list]

                ''' No Hungarian, MLP Only '''
                # pedestrian_list1 = []
                # pre_pedestrian_list1 = []
                # for idx2 in range(z2.shape[1]):
                #     for idx1 in range(z1.shape[1]):
                #         sim_feat2 = z1[:, idx1, :] * z2[:, idx2, :]
                #         if args.distributed:
                #             pred_logits2 = model.module.vic.regression(sim_feat2.squeeze(0))
                #         else:
                #             pred_logits2 = model.vic.regression(sim_feat2.squeeze(0))
                #         pred_prob2 = F.softmax(pred_logits2, dim=0)
                #         pred_score2, pred_class2 = pred_prob2.max(dim=0)
                #         if pred_class2 == 0:
                #             pedestrian_list1 = np.append(pedestrian_list1, idx2)
                #             pre_pedestrian_list1 = np.append(pre_pedestrian_list1, idx1)
                #             break
                # inflow_idx_list1 = [i for i in range(len(pos)) if i not in pedestrian_list1]
                # outflow_idx_list1 = [i for i in range(len(pre_pos)) if i not in pre_pedestrian_list1]


                ''' einsum '''
                sim_feats = torch.einsum('bnc,bmc->bnmc', z2, z1)  # [1, n, m, c]
                sim_feats = sim_feats.view(1, -1, z1.shape[-1])  # [1, n*m, c]
                pos_d = [[float(p[1] / w), float(p[0] / h)] for p in pos]
                pre_pos_d = [[float(pp[1] / w), float(pp[0] / h)] for pp in pre_pos]
                dist, diff = compute_relative_position(torch.tensor(pos_d), torch.tensor(pre_pos_d))
                # dist_feat = pos2posemb2d(diff).cuda()
                dist_feat = model.vic.dist_encoder(diff.cuda().transpose(1, 2)).transpose(1, 2)
                dist_feat = F.normalize(dist_feat, dim=2).contiguous().view(1, -1, 256)
                # sim_feats = torch.cat((sim_feats, dist_feat), dim=2)

                if args.distributed:
                    pred_logits = model.module.vic.regression(sim_feats.squeeze(0), dist_feat)  # [n*m, num_classes]
                else:
                    pred_logits = model.vic.regression(sim_feats.squeeze(0), dist_feat)  # [n*m, num_classes]
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

                # draw(pos, video_name, img_name, inflow_list, cnt_list, len(img_names)-1, cnt_0, len(inflow_idx_list), labels['mask_points'])
                # draw_pair(temp, pos, video_name, pre_img_name, img_name, outflow_list, inflow_list, pre_pedestrian_list, pedestrian_list)
                # draw_failure(pre_pos, pos, video_name, pre_img_name, img_name, pedestrian_list, pre_pedestrian_list, z1, z2)

                z_mask = np.array(outflow_list, dtype = bool)
                # for i in range(3):
                mem = pre_pre_z[0][:len(pre_pos)][z_mask]
                pre_z = [torch.cat((pre_z[0], mem), dim=0)] # cat mem + pre_mem
                pre_pos = np.concatenate((pos, pre_pos[z_mask]), 0)
                # pre_pos=pos
                pre_img_name = img_name

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
            # print(video_name, video_results[video_name]["video_num"],video_results[video_name]["cnt_list"])
            print(video_name, video_results[video_name]["video_num"])
    end = time.time()
    total_length = 0
    for video_name in video_results:
        total_length += video_results[video_name]["frame_num"]
    FPS = total_length / (end - start)
    print("FPS: ", FPS)

    # with open("outputs/json/video_results_test.json", "w") as f:
    #     json.dump(video_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
