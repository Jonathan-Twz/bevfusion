"""
BEV Segmentation Inference Module
=================================

Standalone module for camera-only BEV segmentation inference.
Can be easily integrated into data collection pipelines.

Usage:
    # As a script
    python bev_seg_inference.py --images <path_to_images> --output <output_dir>
    
    # As a module
    from bev_seg_inference import BEVSegmentationInference
    bev_model = BEVSegmentationInference(config_path, checkpoint_path)
    masks = bev_model.infer(images, camera_intrinsics, camera2ego, ...)
"""

import os
import sys
import copy
import argparse
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    """Recursively evaluate config variables."""
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


class BEVSegmentationInference:
    """
    BEV Segmentation Inference Class
    
    This class wraps the BEVFusion model for camera-only BEV segmentation inference.
    It can be easily integrated into any data collection pipeline.
    
    Args:
        config_path: Path to the config file (e.g., configs/nuscenes/seg/camera-bev256d2.yaml)
        checkpoint_path: Path to the model checkpoint (e.g., pretrained/camera-only-seg.pth)
        device: Device to run inference on (default: 'cuda:0')
        map_score_threshold: Threshold for BEV mask (default: 0.5)
    
    Example:
        >>> model = BEVSegmentationInference(
        ...     config_path='configs/nuscenes/seg/camera-bev256d2.yaml',
        ...     checkpoint_path='pretrained/camera-only-seg.pth'
        ... )
        >>> # images: (B, N_cams, C, H, W), camera_intrinsics: (B, N_cams, 4, 4), ...
        >>> masks = model.infer(images, camera_intrinsics, camera2ego, lidar2ego, lidar2camera, lidar2image, camera2lidar)
        >>> # masks: (B, N_classes, H_bev, W_bev) numpy array
    """
    
    # Default map classes (from nuScenes/nuCarla)
    DEFAULT_MAP_CLASSES = [
        'drivable_area',
        'ped_crossing', 
        'walkway',
        'stop_line',
        'carpark_area',
        'divider'
    ]
    
    # Default image normalization (ImageNet)
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        map_score_threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.map_score_threshold = map_score_threshold
        
        # Load config
        configs.load(config_path, recursive=True)
        self.cfg = Config(recursive_eval(configs), filename=config_path)
        
        # Get map classes from config
        self.map_classes = getattr(self.cfg, 'map_classes', self.DEFAULT_MAP_CLASSES)
        self.image_size = getattr(self.cfg, 'image_size', [256, 704])  # H, W
        
        # Build and load model
        self.model = build_model(self.cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Normalization parameters
        self.mean = torch.tensor(self.DEFAULT_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.DEFAULT_STD, device=self.device).view(1, 1, 3, 1, 1)
        
        print(f"[BEVSegmentationInference] Model loaded successfully.")
        print(f"  - Map classes: {self.map_classes}")
        print(f"  - Image size: {self.image_size}")
        print(f"  - Device: {self.device}")
    
    def preprocess_images(
        self,
        images: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess images for inference.
        
        Args:
            images: Input images, shape (B, N_cams, H, W, C) in uint8 BGR format
                    or (B, N_cams, C, H, W) in float32 RGB format [0, 1]
            target_size: Target size (H, W) for resizing. If None, uses config image_size.
        
        Returns:
            Preprocessed images tensor, shape (B, N_cams, C, H, W) normalized
        """
        if target_size is None:
            target_size = self.image_size
        
        # Handle different input formats
        if images.dtype == np.uint8:
            # Convert BGR uint8 to RGB float32 [0, 1]
            if images.ndim == 5 and images.shape[-1] == 3:
                # (B, N, H, W, C) -> (B, N, C, H, W)
                images = images[..., ::-1].copy()  # BGR to RGB
                images = images.transpose(0, 1, 4, 2, 3)  # BHWC to BCHW
            images = images.astype(np.float32) / 255.0
        
        # Convert to tensor
        images = torch.from_numpy(images).to(self.device)
        
        # Resize if needed
        B, N, C, H, W = images.shape
        if (H, W) != tuple(target_size):
            images = images.view(B * N, C, H, W)
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )
            images = images.view(B, N, C, target_size[0], target_size[1])
        
        # Normalize
        images = (images - self.mean) / self.std
        
        return images
    
    def extract_bev_features(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2ego: torch.Tensor,
        lidar2ego: torch.Tensor,
        lidar2camera: torch.Tensor,
        lidar2image: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: Optional[torch.Tensor] = None,
        lidar_aug_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate BEV features without running task heads.

        Returns:
            dict with keys:
                - ``vtransform``: output of LSS / vtransform (after bev_pool), shape
                  typically ``(B, C_vt, H_bev, W_bev)``.
                - ``decoder_neck``: output after ``decoder.backbone`` + ``decoder.neck``,
                  shape typically ``(B, C_neck, H', W')`` (e.g. 256 channels).

        Args:
            Same tensor shapes as :meth:`infer` (batch of multi-view images and calib).
        """
        B, N, C, H, W = images.shape

        images = images.to(self.device)
        camera_intrinsics = camera_intrinsics.to(self.device)
        camera2ego = camera2ego.to(self.device)
        lidar2ego = lidar2ego.to(self.device)
        lidar2camera = lidar2camera.to(self.device)
        lidar2image = lidar2image.to(self.device)
        camera2lidar = camera2lidar.to(self.device)

        if img_aug_matrix is None:
            img_aug_matrix = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        else:
            img_aug_matrix = img_aug_matrix.to(self.device)

        if lidar_aug_matrix is None:
            lidar_aug_matrix = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1)
        else:
            lidar_aug_matrix = lidar_aug_matrix.to(self.device)

        points = [torch.zeros(1, 5, device=self.device) for _ in range(B)]
        metas = [{"token": f"bev_feat_{i}"} for i in range(B)]

        with torch.inference_mode():
            bev_vt = self.model.extract_camera_features(
                images,
                points,
                None,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_depths=None,
            )
            x = self.model.decoder["backbone"](bev_vt)
            bev_neck = self.model.decoder["neck"](x)

        return {"vtransform": bev_vt, "decoder_neck": bev_neck}
    
    def infer(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2ego: torch.Tensor,
        lidar2ego: torch.Tensor,
        lidar2camera: torch.Tensor,
        lidar2image: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: Optional[torch.Tensor] = None,
        lidar_aug_matrix: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Run BEV segmentation inference.
        
        Args:
            images: Preprocessed images, shape (B, N_cams, C, H, W)
            camera_intrinsics: Camera intrinsic matrices, shape (B, N_cams, 4, 4)
            camera2ego: Camera to ego transformation, shape (B, N_cams, 4, 4)
            lidar2ego: LiDAR to ego transformation, shape (B, 4, 4)
            lidar2camera: LiDAR to camera transformation, shape (B, N_cams, 4, 4)
            lidar2image: LiDAR to image projection, shape (B, N_cams, 4, 4)
            camera2lidar: Camera to LiDAR transformation, shape (B, N_cams, 4, 4)
            img_aug_matrix: Image augmentation matrix, shape (B, N_cams, 4, 4). If None, uses identity.
            lidar_aug_matrix: LiDAR augmentation matrix, shape (B, 4, 4). If None, uses identity.
            return_logits: If True, return raw logits instead of thresholded masks.
        
        Returns:
            BEV segmentation masks, shape (B, N_classes, H_bev, W_bev)
            If return_logits=False: bool array with threshold applied
            If return_logits=True: float array with sigmoid probabilities
        """
        B, N, C, H, W = images.shape
        
        # Move inputs to device
        images = images.to(self.device)
        camera_intrinsics = camera_intrinsics.to(self.device)
        camera2ego = camera2ego.to(self.device)
        lidar2ego = lidar2ego.to(self.device)
        lidar2camera = lidar2camera.to(self.device)
        lidar2image = lidar2image.to(self.device)
        camera2lidar = camera2lidar.to(self.device)
        
        # Create identity matrices if augmentation matrices not provided
        if img_aug_matrix is None:
            img_aug_matrix = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        else:
            img_aug_matrix = img_aug_matrix.to(self.device)
            
        if lidar_aug_matrix is None:
            lidar_aug_matrix = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1)
        else:
            lidar_aug_matrix = lidar_aug_matrix.to(self.device)
        
        # Dummy inputs for camera-only model
        points = [torch.zeros(1, 5, device=self.device) for _ in range(B)]
        depths = torch.zeros(B, N, 1, H, W, device=self.device)
        gt_masks_bev = torch.zeros(B, len(self.map_classes), 200, 200, device=self.device)
        
        # Create metas (minimal required info)
        metas = [{'token': f'inference_{i}'} for i in range(B)]
        
        with torch.inference_mode():
            outputs = self.model(
                img=images,
                points=points,
                camera2ego=camera2ego,
                lidar2ego=lidar2ego,
                lidar2camera=lidar2camera,
                lidar2image=lidar2image,
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                metas=metas,
                depths=depths,
                gt_masks_bev=gt_masks_bev,
            )
        
        # Extract masks from outputs
        masks_list = []
        for output in outputs:
            if 'masks_bev' in output:
                masks = output['masks_bev'].numpy()
                if return_logits:
                    masks = torch.sigmoid(torch.from_numpy(masks)).numpy()
                else:
                    masks = masks >= self.map_score_threshold
                masks_list.append(masks)
        
        return np.stack(masks_list, axis=0)
    
    def infer_from_raw(
        self,
        images: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera2ego: np.ndarray,
        lidar2ego: np.ndarray,
        lidar2camera: np.ndarray,
        lidar2image: np.ndarray,
        camera2lidar: np.ndarray,
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Convenience method to run inference from raw numpy arrays.
        
        Args:
            images: Raw images, shape (B, N_cams, H, W, C) in uint8 BGR format
            camera_intrinsics: Camera intrinsic matrices, shape (B, N_cams, 4, 4)
            camera2ego: Camera to ego transformation, shape (B, N_cams, 4, 4)
            lidar2ego: LiDAR to ego transformation, shape (B, 4, 4)
            lidar2camera: LiDAR to camera transformation, shape (B, N_cams, 4, 4)
            lidar2image: LiDAR to image projection, shape (B, N_cams, 4, 4)
            camera2lidar: Camera to LiDAR transformation, shape (B, N_cams, 4, 4)
            return_logits: If True, return raw logits instead of thresholded masks.
        
        Returns:
            BEV segmentation masks, shape (B, N_classes, H_bev, W_bev)
        """
        # Preprocess images
        images_tensor = self.preprocess_images(images)
        
        # Convert numpy arrays to tensors
        camera_intrinsics = torch.from_numpy(camera_intrinsics.astype(np.float32))
        camera2ego = torch.from_numpy(camera2ego.astype(np.float32))
        lidar2ego = torch.from_numpy(lidar2ego.astype(np.float32))
        lidar2camera = torch.from_numpy(lidar2camera.astype(np.float32))
        lidar2image = torch.from_numpy(lidar2image.astype(np.float32))
        camera2lidar = torch.from_numpy(camera2lidar.astype(np.float32))
        
        return self.infer(
            images_tensor,
            camera_intrinsics,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera2lidar,
            return_logits=return_logits,
        )
    
    def visualize_masks(
        self,
        masks: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize BEV segmentation masks.
        
        Args:
            masks: BEV masks, shape (N_classes, H, W) bool or float
            save_path: If provided, save the visualization to this path.
        
        Returns:
            Visualization image, shape (H, W, 3) uint8
        """
        # Color map for different classes
        colors = [
            [255, 179, 0],    # drivable_area - amber
            [128, 62, 117],   # ped_crossing - purple
            [255, 104, 0],    # walkway - orange
            [166, 189, 215],  # stop_line - light blue
            [193, 0, 32],     # carpark_area - red
            [0, 125, 52],     # divider - green
        ]
        
        if masks.dtype == bool:
            masks = masks.astype(np.float32)
        
        n_classes, H, W = masks.shape
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        
        for i in range(min(n_classes, len(colors))):
            mask = masks[i]
            color = np.array(colors[i]) / 255.0
            canvas += mask[:, :, None] * color[None, None, :]
        
        canvas = np.clip(canvas, 0, 1)
        canvas = (canvas * 255).astype(np.uint8)
        
        if save_path is not None:
            cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        return canvas
    
    @property
    def num_classes(self) -> int:
        return len(self.map_classes)
    
    @property
    def class_names(self) -> List[str]:
        return self.map_classes


def main():
    """Command-line interface for BEV segmentation inference."""
    parser = argparse.ArgumentParser(description='BEV Segmentation Inference')
    parser.add_argument('--config', type=str, 
                        default='configs/nuscenes/seg/camera-bev256d2.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                        default='pretrained/camera-only-seg.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to directory containing camera images')
    parser.add_argument('--output', type=str, default='bev_output',
                        help='Output directory for BEV masks')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for BEV mask')
    args = parser.parse_args()
    
    # Initialize model
    model = BEVSegmentationInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        map_score_threshold=args.threshold,
    )
    
    print(f"\n[INFO] BEV Segmentation Inference initialized.")
    print(f"  - Config: {args.config}")
    print(f"  - Checkpoint: {args.checkpoint}")
    print(f"  - Classes: {model.class_names}")
    print(f"\nTo integrate into your pipeline, use:")
    print(f"  from bev_seg_inference import BEVSegmentationInference")
    print(f"  model = BEVSegmentationInference(config_path, checkpoint_path)")
    print(f"  masks = model.infer_from_raw(images, camera_intrinsics, camera2ego, ...)")


if __name__ == '__main__':
    main()
