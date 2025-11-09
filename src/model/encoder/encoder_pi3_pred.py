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
from .pi3.models.pi3 import Pi3, LinearPts3d
from .pi3.models.layers.transformer_head import TransformerDecoder
from .pi3.utils.geometry import se3_inverse, homogenize_points
from .utils import intrinsics_from_focal_center, recover_focal_shift, project_point_clouds
TO_PIL_IMAGE = transforms.ToPILImage()

debug = False

@dataclass
class GroundCamera:
    intrinsics: Float[Tensor, "3 3"]
    angle_radians: float
    angle_degrees: float
    meter_x: float # meter
    meter_y: float # meter
    width: float  # 摄像机图像宽度
    height: float  # 摄像机图像高度


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderPi3PredCfg:
    name: Literal["pi3_pred"]
    gaussians_per_pixel: int
    pose_free: bool = True
    pretrained_weights: str = ""


def world_to_camera_broadcast(pts_all, extrinsics_c2w):
    """
    使用广播将世界坐标系点云批量转换到相机坐标系下。
    
    参数:
    pts_all: 世界坐标系点云, 形状 [b, v, N, gpv, 3]
    extrinsics_c2w: 相机到世界的外参, 形状 [b, 4, 4]
    
    返回:
    pts_cam: 相机坐标系点云, 形状 [b, v, N, gpv, 3]
    """
    
    # 1. 获取 w2c = (c2w)^-1
    b = pts_all.shape[0]
    extrinsics_w2c = torch.inverse(extrinsics_c2w)
    
    # 2. 提取 R (旋转) 和 t (平移)
    R = extrinsics_w2c[:, :3, :3] # [b, 3, 3]
    t = extrinsics_w2c[:, :3, 3:4] # [b, 3, 1]

    # 3. 准备 R, t 和 pts_all 以进行广播
    
    # 在 R 的 v, N, gpv 维度上添加 '1' 
    # [b, 3, 3] -> [b, 1, 1, 1, 3, 3]
    R_expanded = R.view(b, 1, 1, 1, 3, 3)
    
    # 在 t 的 v, N, gpv 维度上添加 '1' 
    # [b, 3, 1] -> [b, 1, 1, 1, 3, 1]
    t_expanded = t.view(b, 1, 1, 1, 3, 1)
    
    # 在 pts_all 的末尾添加一个 '1' 维度，使其成为 "列向量"
    # [b, v, N, gpv, 3] -> [b, v, N, gpv, 3, 1]
    pts_vec = pts_all.unsqueeze(-1)
    
    # 4. 应用变换: P_cam = R @ P_world + t
    # matmul 会自动广播 R_expanded 的 [1,1,1] 维度
    # [b, v, N, gpv, 3, 3] @ [b, v, N, gpv, 3, 1] -> [b, v, N, gpv, 3, 1]
    pts_rotated_vec = R_expanded @ pts_vec
    
    # 广播 t_expanded 并相加
    # [b, v, N, gpv, 3, 1] + [b, 1, 1, 1, 3, 1] -> [b, v, N, gpv, 3, 1]
    pts_cam_vec = pts_rotated_vec + t_expanded
    
    # 5. 移除最后的 '1' 维度
    # [b, v, N, gpv, 3, 1] -> [b, v, N, gpv, 3]
    pts_cam = pts_cam_vec.squeeze(-1)
    
    return pts_cam


class EncoderPi3Pred(Encoder[EncoderPi3PredCfg]):

    def __init__(self, cfg: EncoderPi3PredCfg) -> None:
        super().__init__(cfg)

        self.backbone = Pi3.from_pretrained("yyfz233/Pi3")
        self.backbone.eval()
        self.pose_free = cfg.pose_free

        self.patch_size = self.backbone.point_head.patch_size
        self.gpv = cfg.gaussians_per_pixel

        self.gaussian_decoder = TransformerDecoder(
            in_dim=2*self.backbone.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.backbone.rope,
        )
        self.gaussian_param_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=11 * self.gpv)
        
        self.pos_act = nn.Tanh()
        self.scale_act = nn.Sigmoid()
        self.opacity_act = nn.Sigmoid()
        self.rot_act = lambda x: F.normalize(x, dim=-1)

        self.offset_max = [0.01] * 3
        self.scale_max = [0.001] * 3

        self.meter_per_pixel = 140 / 128
        print("Freezing backbone Pi3 weights.")
        # 冻结 self.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

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
        mask_pts = context['mask'].unsqueeze(-1) # [b, v, H, W, 1]
        mask_res = F.interpolate(context['mask'][:, 1:], size=feat_size, mode='bilinear', align_corners=False)
        sat_img = batch['sat']['sat_pi3'].unsqueeze(1)  # [b, 3, H, W]
        # 使用 torch.no_grad() 上下文管理器
        with torch.no_grad():
            # 在这个代码块中，所有操作都不会被追踪梯度
            # Encode the context images.
            results = self.backbone(torch.cat((sat_img, context["image"]), dim=1))
        
            pts_all = results['points'] # [b, v, h, w, 3]
            conf_all = torch.sigmoid(results['conf'])  # [b, v, h, w, 1]

            # Reconstruct point cloud in reference camera coordinate system
            reference_cam = results['camera_poses'][:, 0]  # [b, 4, 4]
            camera_poses = torch.einsum('bij, bvjk -> bvik', se3_inverse(reference_cam), results['camera_poses'])  # [b, v, 4, 4]
            pts_all = torch.einsum('bij, bvhwj -> bvhwi', se3_inverse(reference_cam), homogenize_points(pts_all))[..., :3]  # [b, v, h, w, 3]
            
            conf_flat = conf_all.view(conf_all.shape[0], -1)  # (B, N)
            conf_threshold = torch.quantile(conf_flat, 0.2, dim=1, keepdim=True)  # Find 20th percentile for each batch
            conf_mask = conf_all >= conf_threshold.view(-1, 1, 1, 1, 1)  # (B, V, H, W, 1)

            pts_all = pts_all * conf_mask.float() * mask_pts.float()
            dis_all = torch.norm(pts_all, dim=-1).reshape(b, -1)  # (B, N)
            norm_factor = dis_all.sum(dim=[-1]) / ((conf_mask.float() * mask_pts.float()).reshape(b, -1).sum(dim=[-1]) + 1e-5) # [B]

            pts_all = pts_all / (norm_factor.view(b, 1, 1, 1, 1) + 1e-5)
            camera_poses[:, :, :3, 3] = camera_poses[:, :, :3, 3] / (norm_factor.view(b, 1, 1) + 1e-5)
            pts_sat = pts_all[:, 0]  # [b, H, W, 3]

            pts_all = F.interpolate(
                rearrange(pts_all, "b v h w c -> (b v) c h w"),
                size=feat_size,
                mode='bilinear',
                align_corners=False,
            )
            pts_all = rearrange(pts_all, "(b v) c h w -> b v h w c", b=b, v=v+1)  # [b, v, h', w', 3]
            pts_all = pts_all[:, 1:]  # 去掉卫星视角点云 [b, v, h', w', 3]

            hidden = results['hidden'][b:]  # [(b v), n, d]
            pos = results['pos'][b:]  # [(b v), n, d]

        with torch.cuda.amp.autocast(enabled=False):
            # for the 3DGS heads
            gaussians_hidden = self.gaussian_decoder(hidden, xpos=pos).float()  # [b, v, n, d]
            gaussians = self.gaussian_param_head(
                [gaussians_hidden[:, self.backbone.patch_start_idx:]], (h, w)
            ) # [(b v), h, w, (self.gpv gs_dim)]
            gaussians = F.interpolate(
                gaussians.permute(0, 3, 1, 2),
                size=feat_size,
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)  # [(b v), h', w', (self.gpv gs_dim)]
            gaussians = rearrange(gaussians, "(b v) h w (gpv d) -> b v (h w) gpv d", b=b, v=v, gpv=self.gpv, d=11)

        # min_depth = torch.min(pts_all.masked_fill(~mask.squeeze(-1).bool(), float('inf'))[..., 2], dim=-1).values.clamp(min=0)
        # pts_all[..., 2] = pts_all[..., 2] - min_depth[:,:,None]
        pts_all = pts_all.reshape(b, v, feat_size[0]*feat_size[1], 1, 3)

        rgb1 = context["image"][:,0]
        rgb1 = F.interpolate(rgb1, size=feat_size, mode='bilinear', align_corners=False)
        rgb1 = rearrange(rgb1, "b c h w -> b (h w) 1 c")
        rgb2 = context["image"][:,1]
        rgb2 = F.interpolate(rgb2, size=feat_size, mode='bilinear', align_corners=False)
        rgb2 = rearrange(rgb2, "b c h w -> b (h w) 1 c")
        rgb_all = torch.stack((rgb1, rgb2), dim=1) # [b v (h w) 1 c]

        rgb_all  = rgb_all.repeat(1, 1, 1, self.gpv, 1)  # [b v (h w) gpv c]
        depths = pts_all[..., -1].unsqueeze(-1) # [b v (h w) 1 1]

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
        
        pts_all = pts_all + offset_xyz
        covariances = build_covariance(scales, rotations) # [b v (h w) gpv 3 3]

        # Recover intrinsics from local points
        points = pts_sat
        masks = torch.sigmoid(results["conf"][:, 0, :, :, 0]) > 0.1
        points = points * masks.float().unsqueeze(-1)
        width = torch.amax(points[..., 0], dim=(1,2)) - torch.amin(points[..., 0], dim=(1,2))
        height = torch.amax(points[..., 1], dim=(1,2)) - torch.amin(points[..., 1], dim=(1,2))
        original_height, original_width = points.shape[-3:-1]
        aspect_ratio = original_width / original_height

        # Recover focal length
        focal, shift = recover_focal_shift(points, masks)
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        # fx, fy = torch.tensor([0.9], device=pts_all.device).repeat(b), torch.tensor([1.0], device=pts_all.device).repeat(b)  # normalize focal length to 1.0
        intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)  # [b, 3, 3]
        intrinsics = intrinsics.detach()
        # g2s_direct_proj = project_point_clouds(
        #     pts_all.reshape(b, -1, 3),
        #     rgb_all.reshape(b, -1, 3),
        #     intrinsics,
        #     feat_size[0], feat_size[1],
        # )
        # g2s_img = TO_PIL_IMAGE(g2s_direct_proj[0].cpu())
        # g2s_img.save('g2s_proj.png')

        # s2s_direct_proj = project_point_clouds(
        #     pts_sat.reshape(b, -1, 3), 
        #     sat_img[:,0].permute(0, 2, 3, 1).reshape(b, -1, 3), 
        #     intrinsics, 
        #     feat_size[0], feat_size[1],
        # )
        # s2s_img = TO_PIL_IMAGE(s2s_direct_proj[0].cpu())
        # s2s_img.save('s2s_proj.png')

        # test_img = TO_PIL_IMAGE(sat_img[0,0].cpu())
        # test_img.save('sat_img.png')

        # test_img2 = TO_PIL_IMAGE(context["image"][0,0].cpu())
        # test_img2.save('grd_img.png')

        # pred ground camera position and angle
        ground_cam_pos = camera_poses[:, 1, :3, 3]  # Ground camera is index 1
        ground_cam_rot = camera_poses[:, 1, :3, :3]
        ground_angle_radians = torch.atan2(ground_cam_rot[:, 1, 2], ground_cam_rot[:, 0, 2])
        ground_angle_degrees = torch.rad2deg(ground_angle_radians)

        proj = torch.einsum('bij,bj->bi', intrinsics, ground_cam_pos)
        x_norm = proj[:, 0] / (proj[:, 2] + 1e-6)
        y_norm = proj[:, 1] / (proj[:, 2] + 1e-6)
        pixel_x = x_norm * feat_size[0] - feat_size[0] // 2
        pixel_y = y_norm * feat_size[1] - feat_size[1] // 2

        # 构建外参矩阵：单位旋转矩阵 + 指定平移
        extrinsics = torch.zeros(b, 4, 4, device=device)
        extrinsics[:, :3, :3] = torch.eye(3, device=device)  # 单位旋转矩阵
        extrinsics[:, :3, 3] = torch.stack([
            ground_cam_pos[:, 0],  # x方向平移
            ground_cam_pos[:, 1],  # y方向平移
            torch.zeros_like(ground_cam_pos[:, 2])  # z方向平移为0
        ], dim=1)
        extrinsics[:, 3, 3] = 1.0  # 齐次坐标

        pts_all = world_to_camera_broadcast(pts_all, extrinsics)
        pts_all = pts_all * rearrange(mask_res, 'b v h w -> b v (h w)')[..., None, None] # [b v (h w) gpv 3]

        return Gaussians(
            pts_all.reshape(b, -1, 3),
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
            intrinsics=intrinsics.unsqueeze(1),
            angle_radians=ground_angle_radians,
            angle_degrees=ground_angle_degrees,
            meter_x=pixel_x * self.meter_per_pixel,
            meter_y=pixel_y * self.meter_per_pixel,
            width=width,
            height=height,
        )