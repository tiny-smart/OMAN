import torch
from torch import nn
from torch.nn import functional as F
import math

from scipy.optimize import linear_sum_assignment
from models.transformer.transformer import TransformerEncoder, TransformerEncoderLayer
from .tri_sim_ot_b import GML
from .extra_loss import kl_loss

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


class VIC_Model(nn.Module):
    def __init__(self, backbone, stride=16, num_feature_levels=1, num_channels=768, proj_channel=256, hidden_dim=256,
                 freeze_backbone=False) -> None:
        super().__init__()
        super().__init__()
        ''' backbone '''
        self.backbone = backbone
        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
        self.stride = stride
        self.num_feature_levels = num_feature_levels
        self.num_channels = num_channels

        # input projection
        self.input_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * num_feature_levels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Conv2d(num_channels + 128, proj_channel, 1)
        self.location_projection = nn.Conv1d(256, 128 * 9, 1)

        self.hidden_dim = hidden_dim

        self.pos_embedder = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        ''' transformer '''
        # encoder
        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048,
                                                0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, 6, None)

        # Regression
        self.regression = MLP((hidden_dim+1)*9, hidden_dim, 2, 3)
        self.ot_loss = GML()

    def forward(self, inputs):
        x = inputs["image_pair"]
        ref_points = inputs["ref_pts"][:, :, :inputs["ref_num"], :]
        ind_points1 = inputs["independ_pts0"][:, :inputs["independ_num0"], ...].squeeze(0)
        ind_points2 = inputs["independ_pts1"][:, :inputs["independ_num1"], ...].squeeze(0)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]

        ref_point1 = ref_points[:, 0, ...].squeeze(0)
        ref_point2 = ref_points[:, 1, ...].squeeze(0)

        point1 = torch.cat((ref_point1, ind_points1), dim=0)
        point2 = torch.cat((ref_point2, ind_points2), dim=0)
        position, distance, dist = compute_relative_position(point1, point2)

        z1_lists = []
        z2_lists = []

        for b in range(x1.shape[0]):
            z1_list = []
            z2_list = []
            for pt1 in point1:
                z1 = self.get_crops(x1[b].unsqueeze(0), pt1)
                z1 = F.interpolate(z1, (96, 96))
                z1_list.append(z1)
            for pt2 in point2:
                z2 = self.get_crops(x2[b].unsqueeze(0), pt2)
                z2 = F.interpolate(z2, (96, 96))
                z2_list.append(z2)

            z1 = torch.cat(z1_list, dim=0).cuda()
            z2 = torch.cat(z2_list, dim=0).cuda()
            z1_lists1 = self.backbone(z1)
            z2_lists1 = self.backbone(z2)

            """ Location Feature """
            location_emb1 = pos2posemb2d(point1).cuda()
            location_emb2 = pos2posemb2d(point2).cuda()
            location_feature1 = self.location_projection(location_emb1.transpose(0, 1)).transpose(0, 1)
            location_feature2 = self.location_projection(location_emb2.transpose(0, 1)).transpose(0, 1)
            location_feature1 = location_feature1.view(len(point1), 128, 3, 3)
            location_feature2 = location_feature2.view(len(point2), 128, 3, 3)
            z1 = torch.cat((z1_lists1, location_feature1), dim=1)
            z2 = torch.cat((z2_lists1, location_feature2), dim=1)

            """ Spatial Feature Aggregation """

            z1_lists.append(self.input_proj(z1))
            z2_lists.append(self.input_proj(z2))

        # num*c*w*h -> (num*w*h)*1*c
        z1 = z1_lists[0].flatten(2).unsqueeze(0).permute(1, 3, 0, 2).flatten(0, 1)
        z2 = z2_lists[0].flatten(2).unsqueeze(0).permute(1, 3, 0, 2).flatten(0, 1)
        pos_emd_1 = self.pos_embedder(z1_lists[0].permute(0, 2, 3, 1)).flatten(2).unsqueeze(0).permute(1, 3, 0, 2).flatten(0, 1)
        pos_emd_2 = self.pos_embedder(z2_lists[0].permute(0, 2, 3, 1)).flatten(2).unsqueeze(0).permute(1, 3, 0, 2).flatten(0, 1)
        num, b, c = z1.shape

        z = torch.cat((z1, z2), dim=0)
        pos_embed = torch.cat((pos_emd_1, pos_emd_2), dim=0)

        encoded_feature, attention_map = self.encoder(z, pos=pos_embed)
        encoded_feature1 = encoded_feature[:num]
        encoded_feature2 = encoded_feature[num:]

        attention_map1 = attention_map[:, :num, num:]  # wrong or not?
        attention_map1 = torch.mean(attention_map1, dim=2).transpose(0, 1).unsqueeze(1)
        attention_map2 = attention_map[:, num:, :num]
        attention_map2 = torch.mean(attention_map2, dim=2).transpose(0, 1).unsqueeze(1)

        encoded_feature1 = torch.cat((encoded_feature1, attention_map1), dim=2)
        encoded_feature2 = torch.cat((encoded_feature2, attention_map2), dim=2)

        encoded_feature1 = encoded_feature1.view(len(point1), 1, 257, 9).flatten(2)
        encoded_feature2 = encoded_feature2.view(len(point2), 1, 257, 9).flatten(2)
        z1 = F.normalize(encoded_feature1[:len(ref_point1)], dim=-1)
        z2 = F.normalize(encoded_feature2[:len(ref_point2)], dim=-1)
        y1 = F.normalize(encoded_feature1[len(ref_point1):], dim=-1)
        y2 = F.normalize(encoded_feature2[len(ref_point2):], dim=-1)

        pre_z = torch.cat((z1, y1), dim=0).transpose(0,1)
        cur_z = torch.cat((z2, y2), dim=0).transpose(0,1)
        match_matrix = torch.bmm(pre_z, cur_z.transpose(1, 2))
        C = match_matrix.cpu().detach().numpy()[0]
        row_ind, col_ind = linear_sum_assignment(-C)
        sim_feat = pre_z[:,row_ind,:] * cur_z[:,col_ind,:]
        pred_logits = self.regression(sim_feat.squeeze(0))

        return z1, z2, y1, y2, pred_logits, row_ind, col_ind, position, distance, dist


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

    def loss(self, outputs, labels):
        z1 = outputs[0].transpose(0, 1)
        z2 = outputs[1].transpose(0, 1)
        y1 = outputs[2].transpose(0, 1)
        y2 = outputs[3].transpose(0, 1)
        pred_logits = outputs[4]
        row_ind, col_ind = outputs[5], outputs[6]
        pos, dist = outputs[7], outputs[8]
        distances = outputs[9]

        """ OT Loss """
        loss_dict = self.ot_loss([z1, y1], [z2, y2], dist)

        """ Classify Loss / Matching Loss """
        pred_prob = F.softmax(pred_logits, dim=1)
        pred_score, pred_class = pred_prob.max(dim=1)

        gt_true = torch.ones_like(pred_score)
        weight_cls = torch.ones_like(pred_score)
        distance = torch.zeros_like(pred_score)

        for i in range(len(row_ind)):
            distance[i] = distances[0][row_ind[i]][col_ind[i]]
            if row_ind[i] < z1.shape[1] and col_ind[i] < z1.shape[1] and distance[i] < distances[0][row_ind[i]][row_ind[i]] + 0.2:
                gt_true[i] = 0
            else:
                weight_cls[i] = dist[0][row_ind[i]][col_ind[i]]

        my_loss_cls = -gt_true*torch.log(pred_prob[:,1])-(1-gt_true)*torch.log(pred_prob[:,0])
        my_loss_cls = my_loss_cls.mean()
        loss_dict["loss_cls"] = torch.sum(my_loss_cls).cuda()
        loss_kl = kl_loss(distance, labels)

        loss_dict["loss_kl"] = torch.tensor(loss_kl).cuda()
        loss_dict["scon_cost"] = loss_dict["scon_cost"].squeeze()
        loss_dict["hinge_cost"] = loss_dict["hinge_cost"].squeeze()
        loss_dict["all"] = loss_dict["scon_cost"] + loss_dict["hinge_cost"] * 0.1 + loss_dict["loss_cls"] + loss_dict["loss_kl"]

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

def compute_relative_position(pt1, pt2):
    relative_pos = []
    relative_poses = []
    distance = []
    distances = []
    dists= []
    for i in range(len(pt1)):
        for j in range(len(pt2)):
            if relative_pos == []:
                temp = torch.abs(pt1[i] - pt2[j])
                relative_pos = temp.unsqueeze(0)
                distance = torch.sqrt(torch.square(temp).sum()).unsqueeze(0)
            else:
                temp = torch.abs(pt1[i] - pt2[j])
                relative_pos = torch.cat((relative_pos, temp.unsqueeze(0)), dim=0)
                distance = torch.cat((distance, torch.sqrt(torch.square(temp).sum()).unsqueeze(0)), dim=0)
        if relative_poses == []:
            relative_poses = relative_pos.unsqueeze(0)
            dists = distance.unsqueeze(0)
            distances = torch.relu(distance-0.2).unsqueeze(0)
        else:
            relative_poses = torch.cat((relative_poses, relative_pos.unsqueeze(0)), dim=0)
            dists = torch.cat((dists, distance.unsqueeze(0)), dim = 0)
            distances = torch.cat((distances, torch.relu(distance-0.2).unsqueeze(0)), dim = 0)
        relative_pos = []
    distances = torch.exp(distances)
    return relative_poses, distances.unsqueeze(0), dists.unsqueeze(0)

def build_vic_model(backbone):
    model = VIC_Model(backbone)
    return model

