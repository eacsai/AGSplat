from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision import transforms

from ..types import Gaussians
from .common.gaussians import build_covariance
from .encoder import Encoder
from .pi3.backbone.backbone_dino import BackboneDino


TO_PIL_IMAGE = transforms.ToPILImage()

debug = False

@dataclass
class GroundCamera:
    width: float  # 摄像机图像宽度
    height: float  # 摄像机图像高度


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderPi3Cfg:
    name: Literal["pi3"]
    gaussians_per_pixel: int
    pose_free: bool = True
    pretrained_weights: str = ""


class EncoderPi3(Encoder[EncoderPi3Cfg]):

    def __init__(self, cfg: EncoderPi3Cfg) -> None:
        super().__init__(cfg)

        self.pose_free = cfg.pose_free
        self.gpv = cfg.gaussians_per_pixel

        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out + 3 + 3, 128),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                11 * self.gpv,
            ),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )

        self.pos_act = nn.Tanh()
        self.scale_act = nn.Sigmoid()
        self.opacity_act = nn.Sigmoid()
        self.rot_act = lambda x: F.normalize(x, dim=-1)

        self.offset_max = [0.01] * 3
        self.scale_max = [0.001] * 3

    def forward(
        self,
        feat_size: tuple[int, int],
        batch: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        image = batch["context"]["image"]
        b, v, _, h, w = image.shape
        pts_feat = rearrange(
            batch["context"]['grd_camera']['pts_gd'],
            "b 1 (v h w) xyz -> b v h w xyz",
            v=v, h=h, w=w,
        )
        img_feat = rearrange(
            image,
            "b v c h w -> b v h w c",
        )
        with torch.cuda.amp.autocast(enabled=False):
            features = self.backbone(image)
            h, w = features.shape[-2:]
            features = rearrange(features, "b v c h w -> b v h w c").contiguous()
            features = self.backbone_projection(torch.cat((features, pts_feat, img_feat), dim=-1))
            features = rearrange(features, "b v h w c -> b v c h w").contiguous()

            if self.high_resolution_skip is not None:
                # Add the high-resolution skip connection.
                skip = rearrange(batch["context"]["image"], "b v c h w -> (b v) c h w")
                skip = self.high_resolution_skip(skip)
                features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

            features = rearrange(features, "b v c h w -> b v (h w) c")
            gaussians = self.to_gaussians(features)
            gaussians = gaussians.view(b, v, h * w, self.gpv, -1)  # [b v (h w) gpv _]

        mask = F.interpolate(batch["context"]['mask'], size=feat_size, mode='nearest')
        mask = rearrange(mask, 'b v h w -> b v (h w)')[..., None, None] # [b v (h w) 1 1]

        rgb1 = batch["context"]["image"][:,0]
        rgb1 = F.interpolate(rgb1, size=feat_size, mode='bilinear', align_corners=False)
        rgb1 = rearrange(rgb1, "b c h w -> b (h w) 1 c")
        rgb2 = batch["context"]["image"][:,1]
        rgb2 = F.interpolate(rgb2, size=feat_size, mode='bilinear', align_corners=False)
        rgb2 = rearrange(rgb2, "b c h w -> b (h w) 1 c")
        rgb_all = torch.stack((rgb1, rgb2), dim=1) # [b v (h w) 1 c]
        rgb_all = rgb_all * mask
        rgb_all  = rgb_all.repeat(1, 1, 1, self.gpv, 1)  # [b v (h w) gpv c]

        if debug:
            opacities = torch.ones_like(gaussians[..., 0], device=gaussians.device).float() * 0.5
            scales = torch.ones_like(gaussians[..., 0:3], device=gaussians.device).float() * 0.005
            offset_xyz = torch.zeros_like(gaussians[..., 0:3], device=gaussians.device).float()
            rotations = torch.ones_like(gaussians[..., 3:7], device=gaussians.device).float() * 0.5
        else:
            gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0]
            gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1]
            gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[1]
            offset_xyz = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) # [b v (h w) gpv 3]

            opacities = self.opacity_act(gaussians[..., 3:4]).squeeze(-1).clamp_min(0.5)
            rotations = self.rot_act(gaussians[..., 4:8])

            scale_x = self.scale_act(gaussians[..., 8:9]) * self.scale_max[0]
            scale_y = self.scale_act(gaussians[..., 9:10]) * self.scale_max[1]
            scale_z = self.scale_act(gaussians[..., 10:11]) * self.scale_max[2]
            scales = torch.cat([scale_x, scale_y, scale_z], dim=-1) # [b v (h w) gpv 3]

            # scales = 0.001 * F.softplus(gaussians[..., 8:11])
            scales = scales.clamp_max(0.1)
        
        pts_all = rearrange(
            batch["context"]['grd_camera']['pts_gd'],
            "b 1 n xyz -> b n 1 xyz",
        ).detach()
        offset_xyz = rearrange(
            offset_xyz,
            "b v (h w) gpv xyz -> b (v h w) gpv xyz",
            h=feat_size[0], w=feat_size[1],
        )
        pts_all = pts_all + offset_xyz
        covariances = build_covariance(scales, rotations) # [b v (h w) gpv 3 3]


        return Gaussians(
            rearrange(
                pts_all,
                "b n gpv xyz -> b (n gpv) xyz",
            ),
            rearrange(
                covariances,
                "b v r gpv i j -> b (v r gpv) i j",
            ),
            rearrange(
                rgb_all,
                "b v r gpv c -> b (v r gpv) c",
            ).unsqueeze(-1),
            rearrange(
                opacities,
                "b v r gpv -> b (v r gpv)",
            ),
        ), GroundCamera(
            width=batch["context"]['grd_camera']['width'].squeeze(-1),
            height=batch["context"]['grd_camera']['height'].squeeze(-1),
        )