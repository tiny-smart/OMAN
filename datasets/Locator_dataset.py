import random
import torch
import numpy as np
import torchvision.transforms as standard_transforms
import warnings

warnings.filterwarnings('ignore')


def locator_dataset(inputs, labels, train=False, flip=False):
    # load image and gt points
    if train:
        img = inputs['image_pair'][:, 0:3, :, :].squeeze(0)
        points = inputs['pt_0'][:,:inputs['cnt_0']].squeeze(0).numpy()
        points = np.multiply(points, np.array([img.shape[2], img.shape[1]]))
        points[:, [0, 1]] = points[:, [1, 0]]
        img1, target1 = process_data(img, points, train=train, flip=False)

        img = inputs['image_pair'][:, 3:6, :, :].squeeze(0)
        points = inputs['pt_1'][:,:inputs['cnt_1']].squeeze(0).numpy()
        points = np.multiply(points, np.array([img.shape[2], img.shape[1]]))
        points[:, [0, 1]] = points[:, [1, 0]]
        img2, target2 = process_data(img, points, train=train, flip=False)

        samples = torch.concatenate([img1.unsqueeze(0), img2.unsqueeze(0)], 0)
        targets = (target1, target2)
    else:
        img = inputs['image_pair'][:, 0:3, :, :].squeeze(0)
        points = inputs['pt_0'][:,:inputs['cnt_0']].squeeze(0).numpy()
        points = np.multiply(points, np.array([img.shape[2], img.shape[1]]))
        points[:,[0,1]] = points[:,[1,0]]
        img1, target1 = process_data(img, points, train=train, flip=flip)

        img = inputs['image_pair'][:, 3:6, :, :].squeeze(0)
        points = inputs['pt_1'][:,:inputs['cnt_1']].squeeze(0).numpy()
        points = np.multiply(points, np.array([img.shape[2], img.shape[1]]))
        points[:, [0, 1]] = points[:, [1, 0]]
        img2, target2 = process_data(img, points, train=train, flip=flip)

        samples = torch.concatenate([img1.unsqueeze(0), img2.unsqueeze(0)], 0)
        targets = (target1, target2)

    return samples, targets

def process_data(img, points, transform=None, train=False, flip=False, patch_size = 256):
    if train:
        scale_range = [0.8, 1.2]
        min_size = min(img.shape[1:])
        scale = random.uniform(*scale_range)

        # interpolation
        if scale * min_size > patch_size:
            img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            points *= scale

    # random crop patch
    if train:
       img, points = random_crop(img, points, patch_size=patch_size)

    # random flip
    if random.random() > 0.5 and train and flip:
        img = torch.flip(img, dims=[2])
        points[:, 1] = patch_size - points[:, 1]

    # target
    target = {}
    target['points'] = torch.Tensor(points)
    target['labels'] = torch.ones([points.shape[0]]).long()

    if train:
        density = compute_density(points)
        target['density'] = density

    return img, target

def compute_density(points):
    """
    Compute crowd density:
        - defined as the average nearest distance between ground-truth points
    """
    points_tensor = torch.from_numpy(points.copy())
    dist = torch.cdist(points_tensor, points_tensor, p=2)
    if points_tensor.shape[0] > 1:
        density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
    else:
        density = torch.tensor(999.0).reshape(-1)
    return density


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w

    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points


def trans_dataset(inputs, labels, image_set):
    # transform = standard_transforms.Compose([
    #     standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                   std=[0.229, 0.224, 0.225]),
    # ])

    if image_set == 'train':
        samples, targets = locator_dataset(inputs, labels, train=True, flip=True)
        return samples, targets
    elif image_set == 'val':
        samples, targets = locator_dataset(inputs, labels, train=False)
        return samples, targets
