from __future__ import annotations

import random
from typing import Callable, Optional
from torchvision.datasets import VisionDataset
from PIL import Image
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import json


def transform():
    return A.Compose([
        A.Resize(768, 1024),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def video_transform():
    return A.Compose([
        A.Resize(768, 1024),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def inverse_normalize(img):
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img * 255
    return img


class VideoDataset(VisionDataset):
    def __init__(self, root: str, annotation_dir: str, transforms: Callable[..., Any] | None = None,
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths = os.listdir(root)
        self.video_paths = [os.path.join(root, video_path) for video_path in self.video_paths]
        self.annotations_paths = os.listdir(annotation_dir)
        self.annotations_paths = [os.path.join(annotation_dir, annotation) for annotation in self.annotations_paths]
        self.annotation = {}
        self.videos = []

        for annotation_path in self.annotations_paths:
            file_index = []
            total_cnt = 0
            file_list = os.listdir(annotation_path)
            for i in range(len(file_list)):
                # 把文件名里的数字提出来排序
                split_file_list = file_list[i].rsplit('_',1)
                if len(split_file_list) == 1:
                    prefix = ''
                else:
                    prefix = split_file_list[0]

                idx = file_list[i].split('.')[0].split('_')[-1]
                file_index.append(int(idx))
                idx_len = len(idx)
            file_index = sorted(file_index)
            video_name = annotation_path.split('/')[-1].split('.')[0]

            self.annotation[video_name] = []

            for file in file_index:
                pedestrian_num = 0
                inflow_num = 0
                outflow_num = 0
                mask_flag = 1
                mask_points = []

                if prefix == '':
                    filename = os.path.join(annotation_path, str(file).zfill(idx_len)) + '.json'
                else:
                    filename = os.path.join(annotation_path, prefix + '_' + str(file).zfill(idx_len)) + '.json'

                with open(filename, 'r') as f:
                    label_list = json.load(f)

                    cnt = len(label_list['shapes'])

                    # if mask_points != []:
                    #     for k in range(len(label_list['shapes'])):
                    #         if label_list['shapes'][k]['label'] in ['mask', 'Mask']:
                    #             mask_box = []
                    #             for j in range(len(label_list['shapes'][k]['points'])):
                    #                 mask_box.append(list(map(int, label_list['shapes'][k]['points'][j])))
                    #             mask_points.append(np.array(mask_box))

                    file_name = prefix + '_' + str(file).zfill(idx_len) + '.jpg'
                    width, height = label_list["imageWidth"], label_list["imageHeight"]

                    ''' mask '''
                    # mask_points = []
                    # for k in range(cnt):
                    #     if label_list['shapes'][k]['label'] in ['mask','Mask'] and mask_flag == 1:
                    #         cnt = cnt - 1
                    #         for j in range(len(label_list['shapes'][k]['points'])):
                    #             mask_points.append(list(map(int, label_list['shapes'][k]['points'][j])))
                    #         mask_flag = 0
                    #         break

                    points = []
                    for k in range(cnt):
                        if label_list['shapes'][k]['label'] in ['mask', 'Mask'] and mask_flag == 1:
                            cnt = cnt - 1
                            for j in range(len(label_list['shapes'][k]['points'])):
                                mask_points.append(list(map(int, label_list['shapes'][k]['points'][j])))
                            mask_flag = 0
                        elif  label_list['shapes'][k]['label'] not in ['mask', 'Mask']:
                            points.append(label_list['shapes'][k]['points'])
                    if points != []:
                        pts = np.array(points)
                        pts = np.concatenate((pts[:,:,0] / width, pts[:,:,1] / height), axis=1)
                    else:
                        pts = -1 * np.ones((cnt, 2))

                    num = len(label_list['shapes'])
                    for j in range(num):
                        if label_list['shapes'][j]['label'] in ['pedestrian', 'Pedestrian', 'pedestrain']:
                            pedestrian_num += 1
                        elif label_list['shapes'][j]['label'] == 'inflow' or label_list['shapes'][j][
                            'label'] == 'Inflow':
                            inflow_num += 1
                            total_cnt += 1
                        elif label_list['shapes'][j]['label'] == 'outflow' or label_list['shapes'][j][
                            'label'] == 'Outflow':
                            outflow_num += 1
                        elif label_list['shapes'][j]['label'] == 'both' or label_list['shapes'][j][
                            'label'] == 'Both':
                            inflow_num += 1
                            outflow_num += 1
                            total_cnt += 1

                    ids = -1 * np.ones((cnt, 1))
                    # pts = -1 * np.ones((cnt, 2))
                    bboxes = -1 * np.ones((cnt, 4))
                    pedestrian_pts = []
                    inflow_pts = []
                    outflow_pts = []

                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
                    bboxes[:, 0] = bboxes[:, 0] / width
                    bboxes[:, 1] = bboxes[:, 1] / height
                    bboxes[:, 2] = bboxes[:, 2] / width
                    bboxes[:, 3] = bboxes[:, 3] / height

                    self.annotation[video_name].append(
                        {"file_name": file_name, "height": height, "width": width, "ids": ids, "pts": pts,
                         "bboxes": bboxes, "cnt": cnt, "pedestrian_pts": pedestrian_pts, "inflow_pts": inflow_pts,
                         "outflow_pts": outflow_pts, "mask_points": mask_points, "inflow_num": inflow_num, "total_cnt": total_cnt})
                f.close()

        for video_path in self.video_paths:
            video_name = video_path.split('/')[-1]
            self.videos.append({
                "video_name": video_name,
                "img_names": [],
                "height": self.annotation[video_name][0]["height"],
                "width": self.annotation[video_name][0]["width"],
                "ids": [],
                "pts": [],
                "bboxes": [],
                "cnt": [],
                "pedestrian_pts": [],
                "inflow_pts": [],
                "outflow_pts": [],
                "mask_points": [],
                "inflow_num": [],
                "total_cnt": [],
            })
            for i in range(0, len(self.annotation[video_name])):
                self.videos[-1]["img_names"].append(self.annotation[video_name][i]["file_name"])
                self.videos[-1]["ids"].append(self.annotation[video_name][i]["ids"])
                self.videos[-1]["pts"].append(self.annotation[video_name][i]["pts"])
                self.videos[-1]["bboxes"].append(self.annotation[video_name][i]["bboxes"])
                self.videos[-1]["cnt"].append(self.annotation[video_name][i]["cnt"])
                self.videos[-1]["pedestrian_pts"].append(self.annotation[video_name][i]["pedestrian_pts"])
                self.videos[-1]["inflow_pts"].append(self.annotation[video_name][i]["inflow_pts"])
                self.videos[-1]["outflow_pts"].append(self.annotation[video_name][i]["outflow_pts"])
                self.videos[-1]["mask_points"].append(self.annotation[video_name][i]["mask_points"])
                self.videos[-1]["inflow_num"].append(self.annotation[video_name][i]["inflow_num"])
                self.videos[-1]["total_cnt"].append(self.annotation[video_name][i]["total_cnt"])

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video = self.videos[index]
        video_name = video["video_name"]
        img_names = video["img_names"]
        height = video["height"]
        width = video["width"]
        ids = video["ids"]
        pts = video["pts"]
        bboxes = video["bboxes"]
        cnt = video["cnt"]
        pedestrian_pts = video["pedestrian_pts"]
        inflow_pts = video["inflow_pts"]
        outflow_pts = video["outflow_pts"]
        inflow_num = video['inflow_num']
        total_cnt = video['total_cnt'][-1]


        imgs = []
        for img_name in img_names:
            img_path = os.path.join(self.root, video_name, img_name)
            with open(img_path, 'rb') as fp_path:
                img = np.array(Image.open(fp_path).convert("RGB"))

                """ mask """
                mask_points = video["mask_points"]
                if mask_points[0]:
                    for i in range(len(mask_points)):
                        cv2.fillPoly(img, [np.array(mask_points[i])], (0, 0, 0))

                # import matplotlib.pyplot as plt
                # plt.imshow(img)
                # plt.title("Masked Image")
                # plt.axis('off')
                # plt.show()
                #
                img = self.transforms(image=img)["image"]

                imgs.append(img)
            fp_path.close()
        imgs = torch.stack(imgs, dim=0)
        labels = {
            "h": height,
            "w": width,
            "pts": pts,
            "bboxes": bboxes,
            "cnt": cnt,
            "video_name": video_name,
            "img_names": img_names,
            "pedestrian_pts": pedestrian_pts,
            "inflow_pts": inflow_pts,
            "outflow_pts": outflow_pts,
            "inflow_num": inflow_num,
            "total_cnt": total_cnt,
            "mask_points": mask_points,
        }
        return imgs, labels


class PairDataset(VisionDataset):
    def __init__(self, root: str, annotation_dir: str, max_len: int, transforms: Callable[..., Any] | None = None,
                 transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None,
                 train=True, step=20, interval=1, force_last=False) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths = os.listdir(root)
        self.video_paths = [os.path.join(root, video_path) for video_path in self.video_paths]
        self.annotations_paths = os.listdir(annotation_dir)
        self.annotations_paths = [os.path.join(annotation_dir, annotation) for annotation in self.annotations_paths]
        self.annotation = {}
        self.pairs = []
        self.max_len = max_len
        for annotation_path in self.annotations_paths:
            file_index = []
            mask_points = []
            file_list = os.listdir(annotation_path)
            for i in range(len(file_list)):
                # 把文件名里的数字提出来排序
                split_file_list = file_list[i].rsplit('_', 1)
                if len(split_file_list) == 1:
                    prefix = ''
                else:
                    prefix = split_file_list[0]

                idx = file_list[i].split('.')[0].split("_")[-1]
                file_index.append(int(idx))

            idx_len = len(idx)
            file_index = sorted(file_index)
            total_label = []  # 统计所有文件的label标签数量
            video_name = annotation_path.split('/')[-1].split('.')[0]
            cnt = 0
            i = 0

            self.annotation[video_name] = []

            for file in file_index:
                j = 0
                pedestrian_num = pedestrian_cnt = 0
                inflow_num = inflow_cnt = 0
                outflow_num = outflow_cnt = 0
                both_num = both_cnt = 0

                # filename = path + 'IMG_' + str(file) + '.json'
                # filename = annotation_path + '/img_' + str(file).zfill(4) + '.json'
                if prefix == '':
                    filename = os.path.join(annotation_path, str(file).zfill(idx_len)) + '.json'
                else:
                    filename = os.path.join(annotation_path, prefix + '_' + str(file).zfill(idx_len)) + '.json'

                with open(filename, 'r') as f:
                    label_list = json.load(f)

                cnt = len(label_list['shapes'])

                file_name = prefix + '_' + str(file).zfill(idx_len) + '.jpg'
                width, height = label_list["imageWidth"], label_list["imageHeight"]

                ids = -1 * np.ones((cnt, 1))
                pts = -1 * np.ones((cnt, 2))
                bboxes = -1 * np.ones((cnt, 4))
                pedestrian_pts = []
                inflow_pts = []
                outflow_pts = []
                both_pts = []

                # mask_points = []
                for j in range(len(label_list['shapes'])):
                    pts[j] = [label_list['shapes'][j]['points'][0][0] / width,
                              label_list['shapes'][j]['points'][0][1] / height]
                    bboxes[j] = [pts[j, 0] - 48/ width, pts[j, 1] - 48/ width, pts[j, 0] + 48/ height, pts[j, 1] + 48/ height]
                    if label_list['shapes'][j]['label'] == 'pedestrian' or label_list['shapes'][j]['label'] == 'Pedestrian' or label_list['shapes'][j]['label'] == 'pedestrain':
                        # pedestrian_pts[pedestrian_cnt] = [label_list['shapes'][j]['points'][0][0] / width,
                        #                                      label_list['shapes'][j]['points'][0][1] / height]
                        pedestrian_pts.append(pts[j])
                        pedestrian_cnt += 1
                    elif label_list['shapes'][j]['label'] == 'inflow' or label_list['shapes'][j]['label'] == 'Inflow':
                        # inflow_pts[inflow_cnt] = [label_list['shapes'][j]['points'][0][0] / width,
                        #                           label_list['shapes'][j]['points'][0][1] / height]
                        inflow_pts.append(pts[j])
                        inflow_cnt += 1
                    elif label_list['shapes'][j]['label'] == 'outflow' or label_list['shapes'][j]['label'] == 'Outflow':
                        # outflow_pts[outflow_cnt] = [label_list['shapes'][j]['points'][0][0] / width,
                        #                             label_list['shapes'][j]['points'][0][1] / height]
                        outflow_pts.append(pts[j])
                        outflow_cnt += 1
                    elif label_list['shapes'][j]['label'] == 'both' or label_list['shapes'][j]['label'] == 'Both':
                        both_pts.append(pts[j])
                        both_cnt += 1

                    elif label_list['shapes'][j]['label'] in ['mask','Mask']:
                        ''' mask '''
                        # cnt = cnt - 1
                        # mask_points.append(list(map(int, label_list['shapes'][j]['points'][k])))
                        if mask_points != []:
                            mask_box = []
                            for k in range(len(label_list['shapes'][j]['points'])):
                                mask_box.append(list(map(int, label_list['shapes'][j]['points'][k])))
                            mask_points.append(np.array(mask_box))



                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
                bboxes[:, 0] = bboxes[:, 0] / width
                bboxes[:, 1] = bboxes[:, 1] / height
                bboxes[:, 2] = bboxes[:, 2] / width
                bboxes[:, 3] = bboxes[:, 3] / height

                self.annotation[video_name].append(
                    {"file_name": file_name, "height": height, "width": width, "ids": ids, "pts": pts,
                     "bboxes": bboxes, "cnt": cnt, "pedestrian_pts": pedestrian_pts, "inflow_pts": inflow_pts,
                     "outflow_pts": outflow_pts, 'both_pts': both_pts, 'mask_points': mask_points})
            f.close()
        for video_path in self.video_paths:
            video_name = video_path.split('/')[-1]
            last_step = 0
            for i in range(1, len(self.annotation[video_name]) - step, interval):
                self.pairs.append({
                    "0": self.annotation[video_name][i],
                    "1": self.annotation[video_name][i + step],
                    "video_name": video_name,
                })
                last_step = i + step
            if force_last and last_step < len(self.annotation[video_name]) - 1:
                self.pairs.append({
                    "0": self.annotation[video_name][last_step],
                    "1": self.annotation[video_name][-1],
                    "video_name": video_name,
                })
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def add_noise(self, pts):
        noise = np.random.normal(scale=0.001, size=pts.shape)
        pts = pts + noise
        pts[pts > 1] = 1
        pts[pts < 0] = 0
        return pts

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        img0_path = os.path.join(self.root, pair["video_name"], pair["0"]["file_name"])
        img1_path = os.path.join(self.root, pair["video_name"], pair["1"]["file_name"])
        video_name = pair["video_name"]
        img_name1 = pair["0"]["file_name"]
        img_name2 = pair["1"]["file_name"]
        cnt_0 = pair["0"]["cnt"]
        cnt_1 = pair["1"]["cnt"]
        pt_0 = pair["0"]["pts"]
        pt_1 = pair["1"]["pts"]
        fused_pts_list0 = pair["0"]["pedestrian_pts"] + pair["0"]["inflow_pts"]
        fused_pts_list1 = pair["1"]["pedestrian_pts"] + pair["1"]["outflow_pts"]
        independ0_list = pair["0"]["outflow_pts"] + pair["0"]["both_pts"]
        independ1_list = pair["1"]["inflow_pts"] + pair["1"]["both_pts"]
        # fused_num = len(pair["0"]["inflow_pts"]) + len(pair["0"]["pedestrian_pts"])
        fused_num = len(fused_pts_list0)
        # print(video_name,img_name1,fused_num,len(pair["0"]["pedestrian_pts"]),len(pair["0"]["inflow_pts"]))
        if len(fused_pts_list1) == 0:
            fused_pts_list1 = -1 * np.ones((len(fused_pts_list0), 2))

        if len(fused_pts_list1) != len(fused_pts_list0):
            return self.__getitem__((index + 1) % len(self))


        if self.train:
            pt_0 = self.add_noise(pt_0)
            pt_1 = self.add_noise(pt_1)
        bbox_0 = pair["0"]["bboxes"]
        bbox_1 = pair["1"]["bboxes"]
        id_0 = pair["0"]["ids"]
        id_1 = pair["1"]["ids"]
        if self.train and (pair["0"]["height"] != pair["1"]["height"] or pair["0"]["width"] != pair["1"][
            "width"] or cnt_0 == 0 or cnt_1 == 0):
            print("error")
            print(img1_path)
            print(pair["0"]["height"], pair["1"]["height"], pair["0"]["width"], pair["1"]["width"], cnt_0, cnt_1)
            return self.__getitem__((index + 1) % len(self))
        fp0_path = open(img0_path, 'rb')
        fp1_path = open(img1_path, 'rb')

        """ mask """
        img0 = np.array(Image.open(fp0_path).convert("RGB"))
        img1 = np.array(Image.open(fp1_path).convert("RGB"))
        mask_points0 = pair["0"]["mask_points"]
        mask_points1 = pair["1"]["mask_points"]
        if mask_points0:
            cv2.fillPoly(img0, [np.array(mask_points0)], (0, 0, 0))
        if mask_points1:
            cv2.fillPoly(img1, [np.array(mask_points1)], (0, 0, 0))

        img0 = self.transforms(image=img0)["image"]
        img1 = self.transforms(image=img1)["image"]





        fp0_path.close()
        fp1_path.close()

        if self.train and (fused_num == 0 or len(independ0_list) == 0 or len(independ1_list) == 0):
            return self.__getitem__((index + 1) % len(self))

        independ_pts0 = -1 * np.ones((self.max_len, 2))
        independ_pts0[:len(independ0_list)] = np.array(independ0_list)
        independ_pts1 = -1 * np.ones((self.max_len, 2))
        independ_pts1[:len(independ1_list)] = np.array(independ1_list)
        x = torch.cat([img0, img1], dim=0)
        fused_pts0 = -1 * np.ones((self.max_len, 2))
        fused_pts1 = -1 * np.ones((self.max_len, 2))

        if fused_num > 0:
            fused_pts0[:fused_num] = np.array(fused_pts_list0)
            fused_pts1[:fused_num] = np.array(fused_pts_list1)

        fused_pts1 = torch.from_numpy(fused_pts1).float()
        fused_pts0 = torch.from_numpy(fused_pts0).float()

        # TODO: 不能简单跳过，两种方法：
        # TODO： ~1.将所有点送入PET，随机选取部分点送入VIC    2.随即裁剪一个区域来训练VIC
        max_num = 150
        if self.train and fused_num >= max_num:
            mask = random.sample(range(fused_num), max_num)
            fused_pts0[:max_num] = fused_pts0[mask]
            fused_pts1[:max_num] = fused_pts1[mask]
            fused_num = max_num

        # visualization
        # cv2_img0=inverse_normalize(img0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        # cv2_img1=inverse_normalize(img1).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        # img_pair=np.concatenate([cv2_img0,cv2_img1],axis=1)
        # img_pair=cv2.cvtColor(img_pair,cv2.COLOR_RGB2BGR)
        # for pt0,pt1 in zip(fused_pts0,fused_pts1):
        #     cv2.circle(img_pair,(int(pt0[0]*1280),int(pt0[1]*720)),5,(0,0,255),-1)
        #     cv2.circle(img_pair,(int(pt1[0]*1280)+1280,int(pt1[1]*720)),5,(0,0,255),-1)
        #     cv2.line(img_pair,(int(pt0[0]*1280),int(pt0[1]*720)),(int(pt1[0]*1280)+1280,int(pt1[1]*720)),(0,0,255),2)
        # cv2.imwrite(f"outputs/vision/{video_name}_{img_name1}_{img_name2}.jpg",img_pair)

        ref_pts = torch.stack([fused_pts0, fused_pts1], dim=0)
        labels = {
            "h": pair["0"]["height"],
            "w": pair["0"]["width"],
            "gt_fuse_pts0": fused_pts0,
            "gt_fuse_pts1": fused_pts1,

            "gt_default_num": pair["0"]["cnt"],
            "gt_duplicate_num": pair["1"]["cnt"],
            "gt_fuse_num": fused_num,
            "video_name": video_name,
            "img_name1": img_name1,
            "img_name2": img_name2,

            "cnt_0": cnt_0,
            "cnt_1": cnt_1,
            "pt_0": pt_0,
            "pt_1": pt_1,
        }

        inputs = {
            "h": pair["0"]["height"],
            "w": pair["0"]["width"],
            "image_pair": x,  # 帧配对 x = torch.cat([img0, img1], dim=0)
            "ref_pts": ref_pts,  # 点配对 ref_pts = torch.stack([fused_pts0, fused_pts1], dim=0)
            "ref_num": fused_num,  # shared pedestrian num
            "independ_pts0": torch.from_numpy(independ_pts0).float(),  # outflow/both pt
            "independ_pts1": torch.from_numpy(independ_pts1).float(),  # inflow/both pt
            "independ_num0": len(independ0_list),  # outflow/both num
            "independ_num1": len(independ1_list),  # inflow/both num

            "cnt_0": cnt_0,
            "cnt_1": cnt_1,
            "pt_0": pt_0,
            "pt_1": pt_1,
        }

        return inputs, labels


def build_dataset(root, annotation_dir, max_len, train=False, step=20, interval=1, force_last=False):
    transforms = transform()
    dataset = PairDataset(root, annotation_dir, max_len, transforms=transforms, train=train, step=step,
                          interval=interval, force_last=force_last)
    return dataset


def build_video_dataset(root, annotation_dir):
    dataset = VideoDataset(root, annotation_dir, transforms=video_transform())
    return dataset
