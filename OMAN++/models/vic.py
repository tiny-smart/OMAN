import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import math
import copy
from typing import Optional, List, Dict, Tuple, Set, Union, Iterable, Any

from scipy.optimize import linear_sum_assignment
from models.transformer.transformer import TransformerDecoderLayer, TransformerDecoder, TransformerEncoder, \
    TransformerEncoderLayer
from util.misc import nested_tensor_from_tensor_list
from . import build_backbone_vgg
from .tri_sim_ot_b import GML


def pos2posemb2d(pos, num_pos_feats=128, temperature=1000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=128, temperature=1000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.ones(x.shape[0], x.shape[1], x.shape[2]).cuda().bool()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim // 2 ** i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x

class CoarseToFineMatcher(nn.Module):
    def __init__(self, input_dim=257, hidden_dim=256, output_dim=2):
        super().__init__()
        self.num_patches = 9

        # Coarse stage: Global projection and similarity
        self.coarse_proj = nn.Linear(input_dim, hidden_dim)
        self.coarse_norm = nn.LayerNorm(hidden_dim)

        # Fine stage: Reshape to 3x3 grid, use CNN for local refinement
        self.fine_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            # nn.ReLU()
        )

        # Dist modulation (FiLM-like)
        self.dist_mod_gamma = nn.Linear(256, hidden_dim)
        self.dist_mod_beta = nn.Linear(256, hidden_dim)

        # self.fine_pool = nn.AdaptiveAvgPool2d(1)
        self.fine_pool = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3)

        # Final head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x, dist_feat):
        n = x.shape[0]

        # Coarse stage
        coarse_x = self.coarse_proj(x)  # [n, hidden]
        coarse_x = self.coarse_norm(coarse_x)

        # Reshape to 3x3 for fine stage (assuming x is flattened 9*257)
        # fine_x = x.view(n, self.num_patches, -1)[:, :, :-1]  # Separate sim_feat part if needed
        # fine_x = fine_x.contiguous().view(n, -1, 3, 3)  # [n, channels, 3, 3] - adjust channels
        fine_x = x.view(-1, self.num_patches, 257)[:, :, :-1]   # onnx
        fine_x = fine_x.contiguous().view(-1, 256, 3, 3)  # onnx

        # Fine convolution
        fine_x = self.fine_conv(fine_x)

        # Dist modulation
        # gamma = self.dist_mod_gamma(dist_feat).view(n, -1, 1, 1)
        # beta = self.dist_mod_beta(dist_feat).view(n, -1, 1, 1)
        gamma = self.dist_mod_gamma(dist_feat).view(-1, 256, 1, 1)    # onnx
        beta = self.dist_mod_beta(dist_feat).view(-1, 256, 1, 1)  # onnx
        fine_x = gamma * fine_x + beta

        # Pool and fuse with coarse
        fine_pool = fine_x.mean(dim=[2, 3])  # [n, hidden]
        # fine_pool = self.fine_pool(fine_x)
        fused = coarse_x + fine_pool

        # Head
        logits = self.head(fused)
        return logits


class VIC_Model(nn.Module):
    def __init__(self, backbone, num_feature_levels=1, num_channels=768, proj_channel=256, hidden_dim=256,
                 freeze_backbone=False) -> None:
        super().__init__()
        ''' backbone '''
        self.backbone = backbone
        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
        self.num_feature_levels = num_feature_levels
        self.num_channels = num_channels

        self.dist_encoder = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * num_feature_levels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Linear(hidden_dim * num_feature_levels, hidden_dim),
            # nn.ReLU(),
        )

        self.input_proj = nn.Conv2d(num_channels + 128, proj_channel, 1)
        self.location_projection = nn.Conv1d(256, 128 * 9, 1)

        self.hidden_dim = hidden_dim

        self.pos_embedder = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        ''' transformer '''
        # encoder
        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048,
                                                0.1, "relu", normalize_before=False)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, 6, None)

        # Regression
        self.regression = CoarseToFineMatcher(input_dim=(hidden_dim + 1) * 9, hidden_dim=hidden_dim, output_dim=2)
        self.ot_loss = GML()

    def forward():
        # comming soon
        pass

    def get_crops(self, z, pt, window_size=[32, 32, 32, 64], absolute=False):
        h, w = z.shape[-2], z.shape[-1]
        if absolute:
            x_min = pt[0] - window_size[0]
            x_max = pt[0] + window_size[1]
            y_min = pt[1] - window_size[2]
            y_max = pt[1] + window_size[3]
        else:

            x_min = pt[0] * w - window_size[0]
            x_max = pt[0] * w + window_size[1]
            y_min = pt[1] * h - window_size[2]
            y_max = pt[1] * h + window_size[3]
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        z = z[..., y_min:y_max, x_min:x_max]

        return z

    def batched_get_crops(self, z, pts, window_size=[64, 64, 64, 128], out_size=96):
        B, C, H, W = z.shape
        N = pts.shape[0]
        device = z.device

        # window size: [left, right, top, bottom]
        left, right, top, bottom = window_size
        x_min = (pts[:, 0] * W - left) / (W - 1) * 2 - 1
        x_max = (pts[:, 0] * W + right) / (W - 1) * 2 - 1
        y_min = (pts[:, 1] * H - top) / (H - 1) * 2 - 1
        y_max = (pts[:, 1] * H + bottom) / (H - 1) * 2 - 1

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, out_size),
            torch.linspace(0, 1, out_size),
            indexing="ij"
        )  # [out_size, out_size]

        grid_x = grid_x.to(pts.device)
        grid_y = grid_y.to(pts.device)

        grid_x = grid_x.unsqueeze(0) * (x_max - x_min).unsqueeze(1).unsqueeze(2) + x_min.unsqueeze(1).unsqueeze(2)
        grid_y = grid_y.unsqueeze(0) * (y_max - y_min).unsqueeze(1).unsqueeze(2) + y_min.unsqueeze(1).unsqueeze(2)
        grid = torch.stack((grid_x, grid_y), dim=-1).to(device)  # [N, out_size, out_size, 2]

        z = z.expand(N, -1, -1, -1)

        crops = F.grid_sample(z, grid, align_corners=True, mode='bilinear')
        return crops  # [N, C, out_size, out_size]

    def loss(self, outputs, labels):
        z1 = outputs[0].transpose(0, 1)
        z2 = outputs[1].transpose(0, 1)
        y1 = outputs[2].transpose(0, 1)
        y2 = outputs[3].transpose(0, 1)
        pred_logits = outputs[4]
        row_ind, col_ind = outputs[5], outputs[6]
        dist = outputs[7]
        dist_cost = outputs[8]

        """ OT Loss """
        loss_dict = self.ot_loss([z1, y1], [z2, y2], dist_cost)

        """ Classify Loss / Matching Loss """
        pred_prob = F.softmax(pred_logits, dim=1)
        pred_score, pred_class = pred_prob.max(dim=1)

        gt_true = torch.ones_like(pred_score)
        weight_cls = torch.ones_like(pred_score)
        distance = torch.zeros_like(pred_score)

        for i in range(len(row_ind)):
            distance[i] = dist[row_ind[i]][col_ind[i]]
            if row_ind[i] < z1.shape[1] and col_ind[i] < z1.shape[1]:  # and distance[i] < dist[0][row_ind[i]][row_ind[i]] + 0.2:
                gt_true[i] = 0
            else:
                weight_cls[i] = dist[row_ind[i]][col_ind[i]]

        my_loss_cls = -gt_true * torch.log(pred_prob[:, 1]) - (1 - gt_true) * torch.log(pred_prob[:, 0])
        my_loss_cls = my_loss_cls.mean()
        loss_dict["test"] = torch.tensor(len(torch.argwhere(pred_class == gt_true)) / len(
            pred_class)).cuda()

        loss_dict["loss_cls"] = torch.sum(my_loss_cls).cuda()

        loss_dict["scon_cost"] = loss_dict["scon_cost"].squeeze()
        loss_dict["hinge_cost"] = loss_dict["hinge_cost"].squeeze()
        loss_dict["all"] = loss_dict["scon_cost"] + loss_dict["hinge_cost"] * 0.1 + loss_dict[
            "loss_cls"]  # + loss_dict["loss_kl"]

        return loss_dict

    def get_box(self, z, pt, window_size=[32, 32, 32, 64], absolute=False):
        h, w = z.shape[-2], z.shape[-1]
        if absolute:
            x_min = pt[0] - window_size[0]
            x_max = pt[0] + window_size[1]
            y_min = pt[1] - window_size[2]
            y_max = pt[1] + window_size[3]
        else:
            x_min = pt[0] * w - window_size[0]
            x_max = pt[0] * w + window_size[1]
            y_min = pt[1] * h - window_size[2]
            y_max = pt[1] * h + window_size[3]
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        box = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32).cuda()

        return box


def compute_relative_position(pts0, pts1):
    pts0_expanded = pts0.unsqueeze(1)  # [n, 1, 2]
    pts1_expanded = pts1.unsqueeze(0)  # [1, n, 2]

    diff = pts0_expanded - pts1_expanded  # [n, n, 2]

    squared_dist = (diff ** 2).sum(dim=-1)  # [n, n]

    dist = torch.sqrt(squared_dist)
    return dist, diff


def build_vic_model(backbone):
    model = VIC_Model(backbone)
    return model
