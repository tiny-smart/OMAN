"""
Backbone modules
"""
from typing import List

import torch
import torch.nn.functional as F
from timm import create_model
from timm.models import features
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


class FeatsFusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

class BackboneBase_ConvNext(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        features = [backbone['stages_0'],backbone['stages_1'],backbone['stages_2'],backbone['stages_3']]
        if return_interm_layers:
            if name == 'convnext_small_384_in22ft1k':
                self.backbone = backbone

                C_size_list = [192, 384, 768]

                self.fpn = FeatsFusion(
                    C_size_list[0], C_size_list[1], C_size_list[2],
                    hidden_size=num_channels,
                    out_size=num_channels,
                    out_kernel=3
                )
            else:
                raise NotImplementedError
        else:
            self.body = nn.Sequential(*features[:])

    def forward(self, tensor_list):
        feats = []
        if isinstance(tensor_list, NestedTensor):
            if self.return_interm_layers:
                xs = tensor_list.tensors
                feats = self.backbone(xs)

                # feature fusion
                features_fpn = self.fpn([feats[1], feats[2], feats[3]])
                features_fpn_4x = features_fpn[0]
                features_fpn_8x = features_fpn[1]

                # get tensor mask
                m = tensor_list.mask
                assert m is not None
                mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
                mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

                out: Dict[str, NestedTensor] = {}
                out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
                out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            if self.return_interm_layers:
                xs = tensor_list
                feats = self.backbone(xs)

                out = feats[3]
        return out

class Backbone_ConvNext(BackboneBase_ConvNext):
    def __init__(self, name: str, pretrained: bool, return_interm_layers: bool, out_indices: List[int], train_backbone: bool, pretrained_cfg: str):
        backbone = create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices, pretrained_cfg=pretrained_cfg)
        num_channels = 768
        super().__init__(backbone, num_channels, name, return_interm_layers)
        self.train_backbone = train_backbone
        self.out_indices = out_indices
        if not self.train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)

    @property
    def feature_info(self):
        return features._get_feature_info(self.backbone, out_indices=self.out_indices)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        if isinstance(tensor_list, NestedTensor):
            for name, x in xs.items():
                out[name] = x
                # position encoding
                pos[name] = self[1](x).to(x.tensors.dtype)
            return out, pos
        else:
            return xs

def build_backbone(args):
    if args.backbone == 'convnext':
        position_embedding = build_position_encoding(args)
        pretrained_cfg = create_model('convnext_small_384_in22ft1k').default_cfg
        pretrained_cfg['file'] = 'pretrained/convnext_small_384_in22ft1k.pth'
        print(pretrained_cfg)
        backbone = Backbone_ConvNext("convnext_small_384_in22ft1k", True, True, [0,1,2,3], True, pretrained_cfg=pretrained_cfg)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
    return model