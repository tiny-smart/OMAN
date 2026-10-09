from __future__ import annotations
import csv

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


def transform():
    return A.Compose([
        A.Resize(720, 1280),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def video_transform():
    return A.Compose([
        A.Resize(720, 1280),
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
            with open(annotation_path, 'r') as f:
                video_name = os.path.splitext(os.path.basename(annotation_path))[0]
                reader = csv.reader(f)

                # Initialize frame data structure
                frame_data = {}
                for row in reader:
                    frame_idx = int(row[0])
                    if frame_idx not in frame_data:
                        frame_data[frame_idx] = {
                            "file_name": f"{frame_idx + 1}.jpg",  # Frame index starts at 0, but images start at 1.jpg
                            "height": None,
                            "width": None,
                            "ids": [],
                            "pts": [],
                            "bboxes": [],
                            "cnt": 0
                        }

                    # Get pedestrian data
                    ped_id = int(row[1])
                    x, y, w, h = map(float, row[2:6])

                    # Calculate center point (pt)
                    pt_x = x + w / 2
                    pt_y = y + h / 2

                    # Add to frame data
                    frame_data[frame_idx]["ids"].append(ped_id)
                    frame_data[frame_idx]["pts"].append([pt_x, pt_y])
                    frame_data[frame_idx]["bboxes"].append([x, y, w, h])
                    frame_data[frame_idx]["cnt"] += 1

                # Convert to numpy arrays and normalize
                for frame_idx, data in frame_data.items():
                    # Read first image to get height and width
                    scene_num = video_name.split("_")[-1]
                    scene_name = video_name.split('_')[0] + '_' + video_name.split('_')[1]
                    img_path = os.path.join(r'/data/bjwang/MovingDroneCrowd/frames', scene_name, scene_num, data["file_name"])
                    with Image.open(img_path) as img:
                        width, height = img.size

                    # Convert lists to numpy arrays
                    cnt = data["cnt"]
                    ids = np.array(data["ids"]).reshape(-1, 1) if cnt > 0 else -1 * np.ones((1, 1))
                    pts = np.array(data["pts"]) if cnt > 0 else -1 * np.ones((1, 2))
                    bboxes = np.array(data["bboxes"]) if cnt > 0 else -1 * np.ones((1, 4))

                    # Normalize coordinates
                    if cnt > 0:
                        pts[:, 0] /= width
                        pts[:, 1] /= height
                        bboxes[:, 0] /= width
                        bboxes[:, 1] /= height
                        bboxes[:, 2] /= width
                        bboxes[:, 3] /= height

                    self.annotation.setdefault(video_name, []).append({
                        "file_name": data["file_name"],
                        "height": height,
                        "width": width,
                        "ids": ids,
                        "pts": pts,
                        "bboxes": bboxes,
                        "cnt": cnt
                    })

        # Organize video data
        for video_path in self.video_paths:
            video_name = os.path.basename(video_path)
            if video_name not in self.annotation:
                continue

            self.videos.append({
                "video_name": video_name,
                "img_names": [],
                "height": self.annotation[video_name][0]["height"],
                "width": self.annotation[video_name][0]["width"],
                "ids": [],
                "pts": [],
                "bboxes": [],
                "cnt": [],
                "video_path": video_path
            })

            for frame_data in self.annotation[video_name]:
                self.videos[-1]["img_names"].append(frame_data["file_name"])
                self.videos[-1]["ids"].append(frame_data["ids"])
                self.videos[-1]["pts"].append(frame_data["pts"])
                self.videos[-1]["bboxes"].append(frame_data["bboxes"])
                self.videos[-1]["cnt"].append(frame_data["cnt"])

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

        video_path = video["video_path"]
        imgs = []
        for img_name in img_names:
            img_path = os.path.join(self.root, video_name, img_name)
            img = self.transforms(image=np.array(Image.open(img_path).convert("RGB")))["image"]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        labels = {
            "h": height,
            "w": width,
            "pts": pts,
            "bboxes": bboxes,
            "cnt": cnt,
            "video_name": video_name,
            "img_names": img_names,
            "video_path": video_path
        }
        return imgs, labels


class PairDataset(VisionDataset):
    def __init__(self, root: str, annotation_dir: str, max_len: int, transforms: Callable[..., Any] | None = None,
                 transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None,
                 train=True, step=15, interval=1, force_last=False) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths = os.listdir(root)
        self.video_paths = [os.path.join(root, video_path) for video_path in self.video_paths]
        self.annotations_paths = os.listdir(annotation_dir)
        self.annotations_paths = [os.path.join(annotation_dir, annotation) for annotation in self.annotations_paths]
        self.annotation = {}
        self.pairs = []
        self.max_len = max_len

        for annotation_path in self.annotations_paths:
            with open(annotation_path, 'r') as f:
                video_name = os.path.splitext(os.path.basename(annotation_path))[0]
                reader = csv.reader(f)

                # Initialize frame data structure
                frame_data = {}
                for row in reader:
                    frame_idx = int(row[0])
                    if frame_idx not in frame_data:
                        frame_data[frame_idx] = {
                            "file_name": f"{frame_idx + 1}.jpg",
                            "height": None,
                            "width": None,
                            "ids": [],
                            "pts": [],
                            "bboxes": [],
                            "cnt": 0
                        }

                    # Get pedestrian data
                    ped_id = int(row[1])
                    x, y, w, h = map(float, row[2:6])

                    # Calculate center point (pt)
                    pt_x = x + w / 2
                    pt_y = y + h / 2

                    # Add to frame data
                    frame_data[frame_idx]["ids"].append(ped_id)
                    frame_data[frame_idx]["pts"].append([pt_x, pt_y])
                    frame_data[frame_idx]["bboxes"].append([x, y, w, h])
                    frame_data[frame_idx]["cnt"] += 1

                # Convert to numpy arrays and normalize
                for frame_idx, data in frame_data.items():
                    # Read first image to get height and width
                    scene_num = video_name.split("_")[-1]
                    scene_name = video_name.split('_')[0] + '_' + video_name.split('_')[1]
                    img_path = os.path.join(r'/data/bjwang/MovingDroneCrowd/frames', scene_name, scene_num, data["file_name"])
                    with Image.open(img_path) as img:
                        width, height = img.size

                    # Convert lists to numpy arrays
                    cnt = data["cnt"]
                    ids = np.array(data["ids"]).reshape(-1, 1) if cnt > 0 else -1 * np.ones((max_len, 1))
                    pts = np.array(data["pts"]) if cnt > 0 else -1 * np.ones((max_len, 2))
                    bboxes = np.array(data["bboxes"]) if cnt > 0 else -1 * np.ones((max_len, 4))

                    # Normalize coordinates
                    if cnt > 0:
                        pts[:, 0] /= width
                        pts[:, 1] /= height
                        bboxes[:, 0] /= width
                        bboxes[:, 1] /= height
                        bboxes[:, 2] /= width
                        bboxes[:, 3] /= height

                    self.annotation.setdefault(video_name, []).append({
                        "file_name": data["file_name"],
                        "height": height,
                        "width": width,
                        "ids": ids,
                        "pts": pts,
                        "bboxes": bboxes,
                        "cnt": cnt
                    })

        # Create frame pairs
        for video_path in self.video_paths:
            video_name = os.path.basename(video_path)
            if video_name not in self.annotation:
                continue

            last_step = 0
            frames = self.annotation[video_name]
            for i in range(1, len(frames) - step, interval):
                self.pairs.append({
                    "0": frames[i],
                    "1": frames[i + step],
                    "video_name": video_name,
                })
                last_step = i + step

            if force_last and last_step < len(frames) - 1:
                self.pairs.append({
                    "0": frames[last_step],
                    "1": frames[-1],
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

        pts_0 = pair["0"]["pts"]
        pts_1 = pair["1"]["pts"]

        if self.train:
            pts_0 = self.add_noise(pts_0)
            pts_1 = self.add_noise(pts_1)
        bbox_0 = pair["0"]["bboxes"]
        bbox_1 = pair["1"]["bboxes"]
        id_0 = pair["0"]["ids"]
        id_1 = pair["1"]["ids"]
        if (pair["0"]["height"] != pair["1"]["height"] or pair["0"]["width"] != pair["1"][
            "width"] or cnt_0 == 0 or cnt_1 == 0):
            print("error")
            print(pair["0"]["height"], pair["1"]["height"], pair["0"]["width"], pair["1"]["width"], cnt_0, cnt_1,
                  video_name, img_name2)
            return self.__getitem__((index + 1) % len(self))
        img0 = self.transforms(image=np.array(Image.open(img0_path).convert("RGB")))["image"]
        img1 = self.transforms(image=np.array(Image.open(img1_path).convert("RGB")))["image"]
        fused_pts_list0 = []
        fused_pts_list1 = []
        id_list0 = []
        id_list1 = []
        fused_num = 0
        id0_pt0_dict = {id[0]: pt for id, pt in zip(id_0, pts_0)}
        id1_pt1_dict = {id[0]: pt for id, pt in zip(id_1, pts_1)}
        independ0_list = []
        independ1_list = []

        '''  ROI  '''
        # scale = 0.6
        # img0 = torch.nn.functional.upsample_bilinear(img0.unsqueeze(0), scale_factor=scale).squeeze(0)
        # pts_0 *= scale
        # img1 = torch.nn.functional.upsample_bilinear(img1.unsqueeze(0), scale_factor=scale).squeeze(0)
        # pts_1 *= scale
        # if img0.shape[1] % 128 != 0 or img0.shape[2] % 128 != 0:
        #     scale = img0.shape[2]
        #     img0 = torch.nn.functional.interpolate(img0.unsqueeze(0),[128 * (img0.shape[1] // 128), 128 * (img0.shape[2] // 128)]).squeeze(0)
        #     scale = img0.shape[2] / scale
        #     pts_0 *= scale
        # if img1.shape[1] % 128 != 0 or img1.shape[2] % 128 != 0:
        #     scale = img1.shape[2]
        #     img1 = torch.nn.functional.interpolate(img1.unsqueeze(0),[128 * (img1.shape[1] // 128), 128 * (img1.shape[2] // 128)]).squeeze(0)
        #     scale = img1.shape[2] / scale
        #     pts_1 *= scale

        for pt, id in zip(pts_0, id_0):
            if id in id_1:
                fused_pts_list0.append(pt)
                id_list0.append(id)
                fused_num += 1
                pt1 = id1_pt1_dict[id[0]]
                fused_pts_list1.append(pt1)
                id_list1.append(id)
            else:
                independ0_list.append(pt)
        for pt, id in zip(pts_1, id_1):
            if id not in id_0:
                independ1_list.append(pt)
        if len(independ0_list) == 0:
            independ0_list.append((0.25, 0.25))
        if len(independ1_list) == 0:
            independ1_list.append((0.75, 0.75))

        if self.train and (fused_num == 0 or len(independ0_list) == 0 or len(independ1_list) == 0):
            print("error")
            print(pair["0"]["height"], pair["1"]["height"], pair["0"]["width"], pair["1"]["width"], cnt_0, cnt_1,
                  video_name, img_name2)
            return self.__getitem__((index + 1) % len(self))

        pt_0 = -1 * np.ones((self.max_len, 2))
        pt_1 = -1 * np.ones((self.max_len, 2))
        pt_0[:cnt_0] = pts_0
        pt_1[:cnt_1] = pts_1
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
        max_num = 70
        if self.train and fused_num >= max_num:
            mask = random.sample(range(fused_num), max_num)
            fused_pts0[:max_num] = fused_pts0[mask]
            fused_pts1[:max_num] = fused_pts1[mask]
            fused_num = max_num

        ''' Visualization '''
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

            "ref_pts": ref_pts,
            "ref_num": fused_num,
            "independ_pts0": torch.from_numpy(independ_pts0).float(),
            "independ_pts1": torch.from_numpy(independ_pts1).float(),
            "independ_num0": len(independ0_list),
            "independ_num1": len(independ1_list),
        }

        inputs = {
            "h": pair["0"]["height"],
            "w": pair["0"]["width"],
            "image_pair": x,
            "ref_pts": ref_pts,
            "ref_num": fused_num,
            "independ_pts0": torch.from_numpy(independ_pts0).float(),
            "independ_pts1": torch.from_numpy(independ_pts1).float(),
            "independ_num0": len(independ0_list),
            "independ_num1": len(independ1_list),

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