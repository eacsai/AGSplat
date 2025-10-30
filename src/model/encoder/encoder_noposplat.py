from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter, DebugGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

inf = float('inf')
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
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.backbone.eval()
        self.pose_free = cfg.pose_free
        if debug:
            self.gaussian_adapter = DebugGaussianAdapter(cfg.gaussian_adapter)
        elif self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        print("Freezing backbone and center heads (downstream_head1, downstream_head2)...")
        # 冻结 self.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 冻结 self.downstream_head1 和 self.downstream_head2
        for param in self.downstream_head1.parameters():
            param.requires_grad = False
        for param in self.downstream_head2.parameters():
            param.requires_grad = False

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view2 3DGS

        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        feat_size: tuple[int, int],
        batch: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        context = batch["context"]
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        mask = F.interpolate(context['mask'], size=feat_size, mode='nearest')
        mask = rearrange(mask, 'b v h w -> b v (h w)')[..., None, None]

        # 使用 torch.no_grad() 上下文管理器
        with torch.no_grad():
            # 在这个代码块中，所有操作都不会被追踪梯度
            # Encode the context images.
            dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
            
            # 这里的 .float() 操作也应该在 no_grad 块内
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        with torch.cuda.amp.autocast(enabled=False):
            # for the 3DGS heads
            if self.gs_params_head_type == 'linear':
                GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
            elif self.gs_params_head_type == 'dpt':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            elif self.gs_params_head_type == 'dpt_gs':
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist())
                GS_res1 = F.interpolate(GS_res1, size=feat_size, mode='bilinear', align_corners=False)
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res2 = F.interpolate(GS_res2, size=feat_size, mode='bilinear', align_corners=False)
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

        extrinsics1 = context["extrinsics"][:, 0]  # cam2world [b, 4, 4]
        pts3d1 = res1['pts3d'] # [b, h, w, 3]

        pts3d1 = F.interpolate(pts3d1.permute(0, 3, 1, 2), size=feat_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        # 将点云从相机坐标系转换到世界坐标系（使用转置）
        # 应用外参变换的转置: [b, h, w, 3] @ [b, 3, 3] -> [b, h, w, 3]
        pts3d1 = torch.einsum('bhwi,bij->bhwj', pts3d1, extrinsics1[:, :3, :3].transpose(-2, -1))

        # 重排为后续处理需要的格式
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")

        extrinsics2 = context["extrinsics"][:, 0]  # cam2world [b, 4, 4]
        pts3d2 = res2['pts3d'] # [b, h, w, 3]

        pts3d2 = F.interpolate(pts3d2.permute(0, 3, 1, 2), size=feat_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        # 将点云从相机坐标系转换到世界坐标系（使用转置）
        # 应用外参变换的转置: [b, h, w, 3] @ [b, 3, 3] -> [b, h, w, 3]
        pts3d2 = torch.einsum('bhwi,bij->bhwj', pts3d2, extrinsics2[:, :3, :3].transpose(-2, -1))

        # 重排为后续处理需要的格式
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)

        # min_depth = torch.min(pts_all.masked_fill(~mask.squeeze(-1).bool(), float('inf'))[..., 2], dim=-1).values.clamp(min=0)
        # pts_all[..., 2] = pts_all[..., 2] - min_depth[:,:,None]
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
        pts_all = pts_all * mask

        rgb1 = view1['img']
        rgb1 = F.interpolate(rgb1, size=feat_size, mode='bilinear', align_corners=False)
        rgb1 = rearrange(rgb1, "b c h w -> b (h w) c")
        rgb2 = view2['img']
        rgb2 = F.interpolate(rgb2, size=feat_size, mode='bilinear', align_corners=False)
        rgb2 = rearrange(rgb2, "b c h w -> b (h w) c")
        rgb_all = torch.stack((rgb1, rgb2), dim=1) # [b v (h w) c]
        rgb_all = (rgb_all.unsqueeze(-2) + 1.0) / 2.0 # [b v (h w) 1 c]
        rgb_all = rgb_all * mask
        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1).clamp_min(0.5)

        # Convert the features and depths into Gaussians.
        if debug:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                rgb_all.unsqueeze(-2),
            )
        elif self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        return Gaussians(
            rearrange(
                context['grd_camera']['pts_gd'].detach(),
                "b v r xyz -> b (v r) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                rgb_all.detach(),
                "b v r 1 c -> b (v r) c 1",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ), GroundCamera(
            width=context['grd_camera']['width'].squeeze(-1).detach(),
            height=context['grd_camera']['height'].squeeze(-1).detach(),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
