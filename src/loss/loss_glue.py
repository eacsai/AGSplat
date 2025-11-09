from dataclasses import dataclass
import torch.nn.functional as F
import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
import numpy as np
import cv2
import os
from torchvision.transforms import ToPILImage
import torch.nn as nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

grid_size = 210.0  # meter
down_sample = 4
def create_metric_grid(grid_size, res, batch_size, only_front=False):
    if only_front: 
        x = np.linspace(-grid_size/2, 0, int(np.floor(res/2))+1)
    else:
        x = np.linspace(-grid_size/2, grid_size/2, res)
    y = np.linspace(-grid_size/2, grid_size/2, res)
    metric_x, metric_y = np.meshgrid(x, y, indexing='ij')
    metric_x, metric_y = torch.tensor(metric_x).flatten().unsqueeze(0).unsqueeze(-1), torch.tensor(metric_y).flatten().unsqueeze(0).unsqueeze(-1)
    metric_coord = torch.cat((metric_x, metric_y), -1).float()
    return metric_coord.repeat(batch_size, 1, 1)

def weighted_procrustes_2d(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):

    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        A_mean, B_mean = (w_norm * A).sum(1, keepdim=True), (w_norm * B).sum(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean, B_mean = A.mean(1, keepdim=True), B.mean(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, False

    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)
    t = B_mean - A_mean @ R.transpose(1, 2)

    return R, t, True

def compute_vce_loss(X0, Rgt, tgt, R, t):
    """
    Computes Virtual Correspondence Error loss between ground-truth and predicted transformations.

    Args:
        X0 (Tensor): Initial 3D coordinates [B, N, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].

    Returns:
        loss (Tensor): Mean reprojection error per batch.
    """
    # Transform points using ground-truth and predicted transformations
    X1_gt = Rgt @ X0.transpose(2, 1) + tgt.transpose(2, 1) 
    X1_pred = R @ X0.transpose(2, 1) + t.transpose(2, 1) 

    # Compute L2 distance 
    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss

metric_coord4loss = create_metric_grid(5.0, 10, 1)

@dataclass
class LossGlueCfg:
    weight: float
    enable_visualization: bool = False
    temperature: float = 0.1
    num_samples_matches: int = 1024

@dataclass
class LossGlueCfgWrapper:
    glue: LossGlueCfg


class LossGlue(Loss[LossGlueCfg, LossGlueCfgWrapper]):

    def __init__(self, cfg: LossGlueCfgWrapper) -> None:
        super().__init__(cfg)

        self.dustbin_score = nn.Parameter(torch.tensor(1.))

    def visualize_camera_position_on_satellite(
        self,
        sat_img,
        gt_positions,
        meter_per_pixel,
        save_path="./camera_position_visualization",
        img_name="camera_pos"
    ):
        """
        在卫星图上可视化相机位置

        Args:
            sat_img: 卫星图像 [C, H, W] 或 [H, W, C]
            gt_positions: 相机真实位置 [[x1, y1], [x2, y2], ...] (单位：米)
            meter_per_pixel: 每像素代表的米数
            save_path: 保存路径
            img_name: 图片名称
        """
        try:
            # 创建保存目录
            os.makedirs(save_path, exist_ok=True)

            # 处理卫星图像格式
            if isinstance(sat_img, torch.Tensor):
                sat_img_np = sat_img.detach().cpu().numpy()
                # 如果是CHW格式，转换为HWC
                if sat_img_np.shape[0] < sat_img_np.shape[2]:
                    sat_img_np = np.transpose(sat_img_np, (1, 2, 0))
            else:
                sat_img_np = sat_img

            # 归一化到0-255范围
            if sat_img_np.max() <= 1.0:
                sat_img_np = (sat_img_np * 255).astype(np.uint8)
            else:
                sat_img_np = sat_img_np.astype(np.uint8)

            # 转换为BGR格式（OpenCV格式）
            if sat_img_np.shape[2] == 3:
                vis_img = cv2.cvtColor(sat_img_np, cv2.COLOR_RGB2BGR)
            else:
                vis_img = sat_img_np.copy()

            H, W = vis_img.shape[:2]
            center_x, center_y = W // 2, H // 2

            # 为每个相机位置绘制三角形
            for i, (x, y) in enumerate(gt_positions):
                # 将米转换为像素
                pixel_x = int(center_x + x / meter_per_pixel)
                pixel_y = int(center_y + y / meter_per_pixel)  # 注意Y轴方向

                # 定义三角形的三个顶点（相对于中心点）
                triangle_size = 15  # 三角形大小
                triangle_points = np.array([
                    [pixel_x, pixel_y - triangle_size],  # 顶点
                    [pixel_x - triangle_size//2, pixel_y + triangle_size//2],  # 左下
                    [pixel_x + triangle_size//2, pixel_y + triangle_size//2]   # 右下
                ], dtype=np.int32)

                # 绘制填充的三角形（红色）
                cv2.fillPoly(vis_img, [triangle_points], (0, 0, 255))  # BGR格式的红色

                # 绘制三角形边框（白色，更粗的线条）
                cv2.polylines(vis_img, [triangle_points], True, (255, 255, 255), 2)

                # 在三角形旁边添加编号
                cv2.putText(vis_img, str(i+1), (pixel_x + triangle_size, pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 在图像上添加标题
            cv2.putText(vis_img, "Camera Positions on Satellite View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 保存图像
            save_file = os.path.join(save_path, f"{img_name}.png")
            cv2.imwrite(save_file, vis_img)
            # print(f"Camera position visualization saved to: {save_file}")

            return save_file

        except Exception as e:
            print(f"Error in camera position visualization: {e}")
            return None

    def forward(
        self,
        batch,
        sat_feat,
        grd_feat,
        meter_per_pixel,
        shift_range_lon=20.0,
        shift_range_lat=20.0,
        weakly_supervised: bool = False,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.

        grd_desc = F.interpolate(grd_feat, scale_factor=0.5, mode='bilinear', align_corners=False).flatten(2)  # [B, C, H, W] -> [B, C, H*W]
        sat_desc = F.interpolate(sat_feat, scale_factor=0.5, mode='bilinear', align_corners=False).flatten(2)  # [B, C, H, W] -> [B, C, H*W]

        matching_score_original = torch.matmul(sat_desc.transpose(1, 2), grd_desc) / self.cfg.temperature  # [B, H*W, H*W]
        matching_score_original[matching_score_original == 0] = float('-inf')
        bs, m, n = matching_score_original.shape

        bins0 = self.dustbin_score.expand(bs, m, 1)
        bins1 = self.dustbin_score.expand(bs, 1, n)
        alpha = self.dustbin_score.expand(bs, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        couplings = F.softmax(couplings, 1) * F.softmax(couplings, 2)
        matching_score = couplings[:, :-1, :-1]

        _, num_kpts_sat, num_kpts_grd = matching_score.shape
        matches_row = matching_score.flatten(1)
        batch_idx = torch.tile(torch.arange(bs).view(bs, 1), [1, self.cfg.num_samples_matches]).reshape(bs, self.cfg.num_samples_matches)
        sampled_idx = torch.multinomial(matches_row, self.cfg.num_samples_matches)

        sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
        sampled_idx_grd = (sampled_idx % num_kpts_grd)

        sat_metric_coord = create_metric_grid(grid_size, sat_feat.shape[-1] // 2, bs).to(sat_feat.device)
        grd_metric_coord = create_metric_grid(grid_size, grd_feat.shape[-1] // 2, bs).to(sat_feat.device)
        
        X = sat_metric_coord[batch_idx, sampled_idx_sat, :]
        Y = grd_metric_coord[batch_idx, sampled_idx_grd, :]
        weights = matches_row[batch_idx, sampled_idx]
        
        R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights) 
        
        if t is None:
            print('t is None')
            return torch.tensor(1.0, device=sat_feat.device)

        loss_vce = compute_vce_loss(metric_coord4loss.to(sat_feat.device), batch["sat"]["gt_r"], batch['sat']["gt_loc"], R, t)
        avg_loss = loss_vce.mean()

        # 可视化第一章卫星图像上相机位置
        # gt_delta_x_rot正数的时候向右移动，负数的时候向左移动
        # gt_delta_y_rot正数的时候向下移动，负数的时候向上移动
        gt_delta_x = batch['sat']['gt_shift_u'][:,0] * shift_range_lon # m
        gt_delta_y = batch['sat']['gt_shift_v'][:,0] * shift_range_lat # m

        pred_u = -t[:,0,1] * meter_per_pixel * down_sample
        pred_v = -t[:,0,0] * meter_per_pixel * down_sample

        # 添加可视化功能
        # if self.cfg.enable_visualization:
        #     # 获取卫星图像（假设batch中有sat_img或可以从sat_feat转换）
        #     sat_img = batch['sat']['sat_ref'][0]  # 取第一个样本的卫星图像

        #     # 使用初始化时设置的meter_per_pixel值

        #     # 准备相机位置列表
        #     camera_positions = []
        #     camera_positions.append([gt_delta_x[0].item(), gt_delta_y[0].item()])
        #     camera_positions.append([pred_u[0].item(), pred_v[0].item()])
        #     # 调用可视化函数
        #     self.visualize_camera_position_on_satellite(
        #         sat_img=sat_img,
        #         gt_positions=camera_positions,
        #         meter_per_pixel=meter_per_pixel,
        #         save_path="./camera_position_visualization",
        #         img_name="camera_pos"
        #     )

        return avg_loss