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

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossBevCfg:
    weight: float
    enable_visualization: bool = False

@dataclass
class LossBevCfgWrapper:
    bev: LossBevCfg


class LossBev(Loss[LossBevCfg, LossBevCfgWrapper]):
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
        corr,
        meter_per_pixel,
        rotation_range=0.0,
        shift_range_lon=20.0,
        shift_range_lat=20.0,
        weakly_supervised: bool = False,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        cos = torch.cos(batch['sat']['gt_heading'][:, 0] * rotation_range / 180 * np.pi)
        sin = torch.sin(batch['sat']['gt_heading'][:, 0] * rotation_range / 180 * np.pi)

        # 可视化第一章卫星图像上相机位置
        # gt_delta_x_rot正数的时候向右移动，负数的时候向左移动
        # gt_delta_y_rot正数的时候向下移动，负数的时候向上移动
        gt_delta_x = batch['sat']['gt_shift_u'][:,0] * shift_range_lon
        gt_delta_y = batch['sat']['gt_shift_v'][:,0] * shift_range_lat

        corr_H, corr_W = corr.shape[-2:]
        B = corr.shape[0]
        if weakly_supervised:
            max_index = torch.argmin(corr[0,0].reshape(-1)).data.cpu().numpy()
        else:
            max_index = torch.argmin(corr[0].reshape(-1)).data.cpu().numpy()
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel

        # 添加可视化功能
        if self.cfg.enable_visualization:
            # 获取卫星图像（假设batch中有sat_img或可以从sat_feat转换）
            sat_img = batch['sat']['sat'][0]  # 取第一个样本的卫星图像

            # 使用初始化时设置的meter_per_pixel值

            # 准备相机位置列表
            camera_positions = []
            camera_positions.append([gt_delta_x[0].item(), gt_delta_y[0].item()])
            camera_positions.append([pred_u.item(), pred_v.item()])
            # 调用可视化函数
            self.visualize_camera_position_on_satellite(
                sat_img=sat_img,
                gt_positions=camera_positions,
                meter_per_pixel=meter_per_pixel / 4,
                save_path="./camera_position_visualization",
                img_name="camera_pos"
            )

        # Supervised Loss Computation
        if not weakly_supervised:
            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
        else:
            # Weakly Supervised Loss Computation
            M, N, H, W = corr.shape
            assert M == N
            dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
            pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
            pos_neg = pos.reshape(-1, 1) - dis
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))

        return loss