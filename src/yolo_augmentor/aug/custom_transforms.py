# custom_transforms.py
"""
Custom Augmentation Transforms for YOLO Dataset
Bbox-localized transformations
Clean, modular, and production-ready implementation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import random

from .utils import radial_mask, apply_masked_blend, random_occlusion_mask


class BaseTransform:
    """Base class for all custom transforms."""

    def __init__(self, bbox_prob: float = 1.0):
        # default probability to apply per-bbox (safe default)
        self.bbox_prob = bbox_prob

    def apply(self, image: np.ndarray, bboxes: List[List[float]],
              class_labels: List[int]) -> Optional[Tuple]:
        """
        Apply transformation to image and bboxes.

        Args:
            image: Input image (RGB)
            bboxes: YOLO format bboxes [[x_center, y_center, width, height], ...]
            class_labels: Class IDs

        Returns:
            Tuple of (transformed_image, bboxes, class_labels) or None if failed
        """
        raise NotImplementedError

    @staticmethod
    def yolo_to_pixels(bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert YOLO format to pixel coordinates."""
        h, w = img_shape[:2]
        x_center, y_center, width, height = bbox

        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        return x1, y1, x2, y2

    @staticmethod
    def pixels_to_yolo(x1: int, y1: int, x2: int, y2: int,
                       img_shape: Tuple[int, int]) -> List[float]:
        """Convert pixel coordinates to YOLO format."""
        h, w = img_shape[:2]

        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        return [x_center, y_center, width, height]

    @staticmethod
    def clip_bbox_to_image(x1: int, y1: int, x2: int, y2: int,
                           img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clip bounding box coordinates to image boundaries."""
        h, w = img_shape[:2]
        # ensure integers and sensible clipping for slicing
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w)))
        y2 = int(max(0, min(y2, h)))
        return x1, y1, x2, y2


class BboxShearTransform(BaseTransform):
    """
    Apply shear transformation only within bounding box regions.
    Simulates fast-moving objects with motion trails (e.g., pickle ball).
    """

    def __init__(self,
                 shear_x_range: Tuple[float, float] = (-60, 60),
                 shear_y_range: Tuple[float, float] = (0, 0),
                 direction: str = 'random',
                 bbox_prob: float = 0.8,
                 blend_edges: bool = True,
                 motion_blur_prob: float = 0.5,
                 motion_blur_intensity: str = 'medium',
                 padding_ratio: float = 0.2):
        super().__init__(bbox_prob=bbox_prob)
        self.shear_x_range = shear_x_range
        self.shear_y_range = shear_y_range
        self.direction = direction
        self.blend_edges = blend_edges
        self.motion_blur_prob = motion_blur_prob
        self.padding_ratio = padding_ratio

        # Motion blur kernel sizes based on intensity
        self.blur_kernels = {
            'low': [5, 7],
            'medium': [7, 11],
            'high': [11, 17]
        }
        self.kernel_range = self.blur_kernels.get(motion_blur_intensity, [7, 11])

    def _get_shear_params(self, direction: str) -> Tuple[float, float]:
        """Generate shear parameters based on direction."""
        if direction == 'horizontal':
            shear_x = random.uniform(*self.shear_x_range)
            shear_y = 0
        elif direction == 'vertical':
            shear_x = 0
            shear_y = random.uniform(*self.shear_y_range)
        elif direction == 'diagonal':
            shear_x = random.uniform(*self.shear_x_range)
            # Diagonal uses reduced range for y
            y_range = (self.shear_y_range[0] * 0.6, self.shear_y_range[1] * 0.6)
            shear_y = random.uniform(*y_range)
        else:  # random
            dir_choice = random.choice(['horizontal', 'vertical', 'diagonal'])
            return self._get_shear_params(dir_choice)

        return shear_x, shear_y

    def _apply_shear_to_region(self, region: np.ndarray,
                               shear_x: float, shear_y: float) -> np.ndarray:
        """Apply shear transformation to image region."""
        h, w = region.shape[:2]

        # Convert degrees to shear factors
        shear_x_factor = np.tan(np.radians(shear_x))
        shear_y_factor = np.tan(np.radians(shear_y))

        # Create shear matrix
        M = np.array([
            [1, shear_x_factor, 0],
            [shear_y_factor, 1, 0]
        ], dtype=np.float32)

        # Calculate new dimensions
        new_w = int(w + abs(shear_x_factor * h))
        new_h = int(h + abs(shear_y_factor * w))
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Adjust translation
        M[0, 2] = max(0, shear_x_factor * h) if shear_x_factor > 0 else 0
        M[1, 2] = max(0, shear_y_factor * w) if shear_y_factor > 0 else 0

        # Apply warp
        sheared = cv2.warpAffine(region, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        # Resize back to original dimensions (only if valid)
        if sheared.shape[0] > 0 and sheared.shape[1] > 0:
            sheared = cv2.resize(sheared, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            sheared = region.copy()

        return sheared

    def _apply_motion_blur(self, region: np.ndarray,
                           shear_x: float, shear_y: float) -> np.ndarray:
        """Apply directional motion blur based on shear direction."""
        # ensure odd kernel sizes in range and choose odd
        lo, hi = self.kernel_range
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            lo, hi = hi, lo
        # ensure odd start/end
        lo = lo | 1
        hi = hi | 1
        kernel_size = random.choice(list(range(lo, hi + 1, 2)))

        # Create directional kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        if abs(shear_x) > abs(shear_y):
            # Horizontal motion blur
            kernel[kernel_size // 2, :] = 1.0
        else:
            # Vertical motion blur
            kernel[:, kernel_size // 2] = 1.0

        s = kernel.sum()
        if s == 0:
            # fallback to simple horizontal kernel
            kernel[kernel_size // 2, :] = 1.0
            s = kernel.sum()
        kernel /= s

        # Apply blur
        blurred = cv2.filter2D(region, -1, kernel)

        return blurred

    def _create_blend_mask(self, height: int, width: int,
                           edge_size: int) -> np.ndarray:
        """Create smooth alpha mask for edge blending."""
        mask = np.ones((height, width), dtype=np.float32)

        if edge_size > 0:
            # Top/bottom edge
            if height > edge_size * 2:
                mask[:edge_size, :] *= np.linspace(0, 1, edge_size)[:, np.newaxis]
                mask[-edge_size:, :] *= np.linspace(1, 0, edge_size)[:, np.newaxis]

            # Left/right edge
            if width > edge_size * 2:
                mask[:, :edge_size] *= np.linspace(0, 1, edge_size)[np.newaxis, :]
                mask[:, -edge_size:] *= np.linspace(1, 0, edge_size)[np.newaxis, :]

        return mask

    def apply(self, image: np.ndarray, bboxes: List[List[float]],
              class_labels: List[int]) -> Optional[Tuple]:
        """Apply bbox-localized shear transformation."""
        if not bboxes:
            return image, bboxes, class_labels

        aug_image = image.copy()
        img_h, img_w = image.shape[:2]

        # Determine base direction for all bboxes (for consistency)
        base_direction = self.direction if self.direction != 'random' else random.choice(
            ['horizontal', 'vertical', 'diagonal']
        )

        for i, bbox in enumerate(bboxes):
            # Randomly skip some bboxes
            if random.random() > self.bbox_prob:
                continue

            # Get pixel coordinates (pass img shape correctly)
            x1, y1, x2, y2 = self.yolo_to_pixels(bbox, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))

            if x2 <= x1 or y2 <= y1:
                continue

            # Calculate adaptive padding based on bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            padding_x = int(bbox_width * self.padding_ratio)
            padding_y = int(bbox_height * self.padding_ratio)

            # Apply padding and clip to image boundaries
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(img_w, x2 + padding_x)
            y2_padded = min(img_h, y2 + padding_y)

            if x2_padded <= x1_padded or y2_padded <= y1_padded:
                continue

            # Extract bbox region
            region = aug_image[y1_padded:y2_padded, x1_padded:x2_padded].copy()
            if region.size == 0:
                continue

            # Generate shear parameters
            shear_x, shear_y = self._get_shear_params(base_direction)

            # Apply shear
            sheared_region = self._apply_shear_to_region(region, shear_x, shear_y)

            # Apply motion blur if needed
            if random.random() < self.motion_blur_prob:
                sheared_region = self._apply_motion_blur(sheared_region, shear_x, shear_y)

            # Blend back into image
            if self.blend_edges:
                edge_size = max(1, min(10, (x2_padded - x1_padded) // 10, (y2_padded - y1_padded) // 10))
                mask = self._create_blend_mask(y2_padded - y1_padded, x2_padded - x1_padded, edge_size)

                for c in range(3):
                    aug_image[y1_padded:y2_padded, x1_padded:x2_padded, c] = (
                            mask * sheared_region[:, :, c] +
                            (1 - mask) * aug_image[y1_padded:y2_padded, x1_padded:x2_padded, c]
                    ).astype(np.uint8)
            else:
                aug_image[y1_padded:y2_padded, x1_padded:x2_padded] = sheared_region

        return aug_image, bboxes, class_labels


class BboxOcclusionTransform(BaseTransform):
    """
    Apply realistic occlusions within and around bounding boxes.
    Simulates partial object visibility, overlapping objects, shadows.
    """

    def __init__(self,
                 occlusion_type: str = 'mixed',
                 intensity: str = 'medium',
                 bbox_prob: float = 0.6,
                 realistic_shapes: bool = True,
                 partial_coverage: bool = True,
                 occlusion_color: str = 'auto'):
        super().__init__(bbox_prob=bbox_prob)
        self.occlusion_type = occlusion_type
        self.intensity = intensity
        self.realistic_shapes = realistic_shapes
        self.partial_coverage = partial_coverage
        self.occlusion_color = occlusion_color

        # Coverage percentages based on intensity
        self.coverage_ranges = {
            'low': (0.1, 0.25),
            'medium': (0.15, 0.4),
            'high': (0.25, 0.55)
        }
        self.coverage_range = self.coverage_ranges.get(intensity, (0.2, 0.5))

    def _get_occlusion_color(self, image: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int]:
        """Get occlusion color based on configuration."""
        if self.occlusion_color == 'black':
            return (0, 0, 0)
        elif self.occlusion_color == 'gray':
            gray_val = random.randint(40, 120)
            return (gray_val, gray_val, gray_val)
        else:  # auto - sample from nearby region
            h, w = image.shape[:2]
            sx1 = max(0, x1 - 50)
            sx2 = min(w - 1, x2 + 50)
            sy1 = max(0, y1 - 50)
            sy2 = min(h - 1, y2 + 50)
            if sx2 < sx1:
                sx2 = sx1
            if sy2 < sy1:
                sy2 = sy1
            sample_x = random.randint(sx1, sx2)
            sample_y = random.randint(sy1, sy2)
            color = tuple(image[sample_y, sample_x].tolist())
            # Darken it a bit
            color = tuple(max(0, int(c * 0.7)) for c in color)
            return color

    def _create_geometric_occlusion(self, height: int, width: int,
                                    coverage: float) -> np.ndarray:
        """Create geometric shape occlusion mask."""
        mask = np.zeros((height, width), dtype=np.uint8)

        shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])

        if shape_type == 'rectangle':
            # Random rectangle position
            occ_w = int(width * random.uniform(0.3, 0.8))
            occ_w = max(1, occ_w)
            occ_h = int(height * coverage * random.uniform(0.8, 1.5))
            occ_h = max(1, occ_h)

            x_start = random.randint(0, max(0, width - occ_w))
            y_start = random.randint(0, max(0, height - occ_h))

            mask[y_start:y_start + occ_h, x_start:x_start + occ_w] = 255

        elif shape_type == 'ellipse':
            center_x = random.randint(int(width * 0.2), max(int(width * 0.8), int(width * 0.2)))
            center_y = random.randint(int(height * 0.2), max(int(height * 0.8), int(height * 0.2)))

            axes_x = max(1, int(width * coverage * random.uniform(0.3, 0.6)))
            axes_y = max(1, int(height * coverage * random.uniform(0.3, 0.6)))

            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y),
                        random.randint(0, 180), 0, 360, 255, -1)

        else:  # triangle
            pts = np.array([
                [random.randint(0, max(0, width)), random.randint(0, max(0, height))],
                [random.randint(0, max(0, width)), random.randint(0, max(0, height))],
                [random.randint(0, max(0, width)), random.randint(0, max(0, height))]
            ], dtype=np.int32)

            cv2.fillPoly(mask, [pts], 255)

        return mask

    def _create_organic_occlusion(self, height: int, width: int,
                                  coverage: float) -> np.ndarray:
        """Create organic/irregular shape occlusion mask."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Create multiple overlapping circles for organic shape
        num_circles = random.randint(3, max(3, min(8, width // 10 + height // 10)))

        for _ in range(num_circles):
            center_x = random.randint(0, max(0, width - 1))
            center_y = random.randint(0, max(0, height - 1))
            radius = max(1, int(min(width, height) * coverage * random.uniform(0.2, 0.5)))

            cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Apply Gaussian blur for smooth edges
        kernel_size = max(5, int(min(width, height) * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        # Threshold to binary
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

        return mask

    def _create_shadow_occlusion(self, height: int, width: int,
                                 coverage: float) -> Tuple[np.ndarray, float]:
        """Create shadow-like occlusion mask with alpha."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Create gradient shadow
        shadow_type = random.choice(['linear', 'radial'])

        if shadow_type == 'linear':
            # Vertical or horizontal gradient
            if random.random() < 0.5:
                # Vertical
                gradient = np.linspace(0, 255, max(1, height)).astype(np.uint8)
                mask = np.tile(gradient[:, np.newaxis], (1, width))
            else:
                # Horizontal
                gradient = np.linspace(0, 255, max(1, width)).astype(np.uint8)
                mask = np.tile(gradient, (height, 1))
        else:
            # Radial gradient
            center_x = width // 2
            center_y = height // 2
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2) or 1.0

            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask = ((1 - dist / max_dist) * 255).astype(np.uint8)

        # Shadow is semi-transparent
        alpha = random.uniform(0.2, 0.5)

        return mask, alpha

    def apply(self, image: np.ndarray, bboxes: List[List[float]],
              class_labels: List[int]) -> Optional[Tuple]:
        """Apply bbox-localized occlusion."""
        if not bboxes:
            return image, bboxes, class_labels

        aug_image = image.copy()
        img_h, img_w = image.shape[:2]

        for i, bbox in enumerate(bboxes):
            # Randomly skip some bboxes
            if random.random() > self.bbox_prob:
                continue

            # Get pixel coordinates
            x1, y1, x2, y2 = self.yolo_to_pixels(bbox, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract bbox region
            region = aug_image[y1:y2, x1:x2].copy()
            if region.size == 0:
                continue
            h, w = region.shape[:2]

            # Determine occlusion type
            if self.occlusion_type == 'mixed':
                occ_type = random.choice(['geometric', 'organic', 'shadow'])
            else:
                occ_type = self.occlusion_type

            # Get coverage
            coverage = random.uniform(*self.coverage_range)

            # Create occlusion mask
            if occ_type == 'geometric':
                mask = self._create_geometric_occlusion(h, w, coverage)
                alpha = 1.0
            elif occ_type == 'organic':
                mask = self._create_organic_occlusion(h, w, coverage)
                alpha = 1.0
            else:  # shadow
                mask, alpha = self._create_shadow_occlusion(h, w, coverage)

            # Get occlusion color
            occ_color = self._get_occlusion_color(aug_image, x1, y1, x2, y2)

            # Apply occlusion
            mask_3ch = (mask[:, :, np.newaxis].astype(np.float32) / 255.0)

            for c in range(3):
                region[:, :, c] = (
                        alpha * mask_3ch[:, :, 0] * occ_color[c] +
                        (1 - alpha * mask_3ch[:, :, 0]) * region[:, :, c]
                ).astype(np.uint8)

            aug_image[y1:y2, x1:x2] = region

        return aug_image, bboxes, class_labels


class AdaptiveMotionBlurTransform(BaseTransform):
    """
    Apply motion blur with direction and intensity based on bbox aspect ratio.
    Automatically determines motion direction from object shape.
    """

    def __init__(self,
                 kernel_size_range: Tuple[int, int] = (7, 15),
                 auto_direction: bool = True,
                 intensity: str = 'medium',
                 bbox_prob: float = 0.7):
        super().__init__(bbox_prob=bbox_prob)
        self.kernel_size_range = kernel_size_range
        self.auto_direction = auto_direction
        self.intensity = intensity

        # Intensity multipliers
        self.intensity_multipliers = {
            'low': 0.6,
            'medium': 1.0,
            'high': 1.4
        }
        self.multiplier = self.intensity_multipliers.get(intensity, 1.0)

    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create directional motion blur kernel."""
        size = max(3, int(size))
        if size % 2 == 0:
            size += 1
        kernel = np.zeros((size, size), dtype=np.float32)

        # Create line kernel at specified angle
        center = size // 2

        # Calculate line endpoints
        length = size - 1
        dx = np.cos(np.radians(angle)) * length / 2
        dy = np.sin(np.radians(angle)) * length / 2

        # Draw line robustly
        x1, y1 = int(round(center - dx)), int(round(center - dy))
        x2, y2 = int(round(center + dx)), int(round(center + dy))

        # clip endpoints
        x1 = int(max(0, min(x1, size - 1)))
        x2 = int(max(0, min(x2, size - 1)))
        y1 = int(max(0, min(y1, size - 1)))
        y2 = int(max(0, min(y2, size - 1)))

        cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)

        # Normalize safely
        s = kernel.sum()
        if s == 0:
            # fallback: horizontal line in center
            kernel[center, :] = 1.0
            s = kernel.sum()
        kernel /= s

        return kernel.astype(np.float32)

    def apply(self, image: np.ndarray, bboxes: List[List[float]],
              class_labels: List[int]) -> Optional[Tuple]:
        """Apply adaptive motion blur."""
        if not bboxes:
            return image, bboxes, class_labels

        aug_image = image.copy()
        img_h, img_w = image.shape[:2]

        for i, bbox in enumerate(bboxes):
            if random.random() > self.bbox_prob:
                continue

            # Get pixel coordinates
            x1, y1, x2, y2 = self.yolo_to_pixels(bbox, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract bbox region
            region = aug_image[y1:y2, x1:x2].copy()
            if region.size == 0:
                continue
            h, w = region.shape[:2]

            # Determine blur direction
            if self.auto_direction:
                # Aspect ratio determines direction
                aspect_ratio = w / h if h > 0 else 1.0

                if aspect_ratio > 1.5:
                    # Wide bbox - horizontal motion
                    angle = 0
                elif aspect_ratio < 0.67:
                    # Tall bbox - vertical motion
                    angle = 90
                else:
                    # Roughly square - random diagonal
                    angle = random.choice([45, 135, 225, 315])
            else:
                angle = random.uniform(0, 360)

            # Determine kernel size (inclusive) and apply multiplier
            lo, hi = self.kernel_size_range
            lo, hi = int(lo), int(hi)
            if hi < lo:
                lo, hi = hi, lo
            # choose odd sizes only (step 2)
            candidate_sizes = [s for s in range(max(3, lo | 1), hi + 1, 2)]
            if not candidate_sizes:
                candidate_sizes = [3]
            kernel_size = random.choice(candidate_sizes)
            kernel_size = int(max(3, round(kernel_size * self.multiplier)))
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create and apply motion blur
            kernel = self._create_motion_kernel(kernel_size, angle)
            blurred = cv2.filter2D(region, -1, kernel)

            aug_image[y1:y2, x1:x2] = blurred

        return aug_image, bboxes, class_labels


# Additional utility transforms

class BboxColorJitterTransform(BaseTransform):
    """Apply color jittering only within bbox regions."""

    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 saturation_range: Tuple[float, float] = (0.7, 1.3),
                 bbox_prob: float = 0.6):
        super().__init__(bbox_prob=bbox_prob)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def apply(self, image: np.ndarray, bboxes: List[List[float]],
              class_labels: List[int]) -> Optional[Tuple]:
        """Apply bbox-localized color jittering."""
        if not bboxes:
            return image, bboxes, class_labels

        aug_image = image.copy()
        img_h, img_w = image.shape[:2]

        for bbox in bboxes:
            if random.random() > self.bbox_prob:
                continue

            x1, y1, x2, y2 = self.yolo_to_pixels(bbox, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))

            if x2 <= x1 or y2 <= y1:
                continue

            region = aug_image[y1:y2, x1:x2].copy()
            if region.size == 0:
                continue

            # Apply brightness
            brightness = random.uniform(*self.brightness_range)
            region = np.clip(region.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Apply contrast
            contrast = random.uniform(*self.contrast_range)
            mean = region.mean(axis=(0, 1), keepdims=True)
            region = np.clip((region.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

            # Apply saturation (assumes RGB input, keep as is)
            saturation = random.uniform(*self.saturation_range)
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            region = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            aug_image[y1:y2, x1:x2] = region

        return aug_image, bboxes, class_labels


# ============================================================
# BboxMultiBlurAndShearTransform
# ============================================================
class BboxMultiBlurAndShearTransform(BaseTransform):
    """Applies random blur inside bbox, then aggressive shear."""
    def __init__(self, **params):
        # preserve provided bbox_prob if present
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.blur_types = params.get('blur_types', ['gaussian'])
        self.blur_strength_range = params.get('blur_strength_range', (3, 15))
        # ensure sensible ints
        a, b = int(self.blur_strength_range[0]), int(self.blur_strength_range[1])
        if b < a:
            a, b = b, a
        self.blur_strength_range = (max(1, a), max(1, b))

        self.shear_x_range = params.get('shear_x_range', (-80, 80))
        self.shear_y_range = params.get('shear_y_range', (-40, 40))

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        for i, box in enumerate(bboxes):
            if random.random() > self.bbox_prob:
                continue
            # pass (h, w)
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # Random blur: choose k inclusive and ensure it's odd and >=3
            blur_type = random.choice(self.blur_types)
            k = random.randint(self.blur_strength_range[0], self.blur_strength_range[1])
            k = max(3, k | 1)

            if blur_type == 'gaussian':
                crop = cv2.GaussianBlur(crop, (k, k), 0)
            elif blur_type == 'median':
                # median requires odd kernel
                crop = cv2.medianBlur(crop, k)
            elif blur_type == 'bilateral':
                crop = cv2.bilateralFilter(crop, k, 75, 75)
            elif blur_type == 'defocus':
                kk = max(3, k)
                kernel = np.zeros((kk, kk), np.float32)
                cv2.circle(kernel, (kk // 2, kk // 2), max(1, kk // 3), 1, -1)
                s = kernel.sum()
                if s != 0:
                    kernel /= s
                    crop = cv2.filter2D(crop, -1, kernel)

            # Shear
            sx = random.uniform(*self.shear_x_range)
            sy = random.uniform(*self.shear_y_range)
            M = np.float32([[1, np.tan(np.radians(sx)) / 50, 0],
                            [np.tan(np.radians(sy)) / 50, 1, 0]])
            crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT)
            image[y1:y2, x1:x2] = crop
        return image, bboxes, labels


# ============================================================
# BboxExtremeShearOcclude
# ============================================================
class BboxExtremeShearOcclude(BaseTransform):
    """Extreme shear, brightness, and occlusion near bbox."""
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.shear_x_range = params.get('shear_x_range', (-100, 100))
        self.shear_y_range = params.get('shear_y_range', (-60, 60))
        self.brightness_shift = params.get('brightness_shift', (1.5, 2.5))
        self.occlusion_intensity = params.get('occlusion_intensity', 'medium')

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            sx, sy = random.uniform(*self.shear_x_range), random.uniform(*self.shear_y_range)
            M = np.float32([[1, np.tan(np.radians(sx)) / 50, 0],
                            [np.tan(np.radians(sy)) / 50, 1, 0]])
            crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT)

            alpha = random.uniform(*self.brightness_shift)
            crop = np.clip(crop.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

            mask = random_occlusion_mask(*crop.shape[:2], intensity=self.occlusion_intensity)
            if mask is not None and mask.shape[:2] == crop.shape[:2]:
                crop[mask > 0] = 0
            image[y1:y2, x1:x2] = crop
        return image, bboxes, labels


# ============================================================
# NearBboxExtremeBrighten
# ============================================================
class NearBboxExtremeBrighten(BaseTransform):
    """
    Rectangular bright halo AROUND bbox (not inside).
    Simulates stadium lights, flash reflections, or directional lighting.
    """
    
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        
        # Rectangular expansion ranges (multipliers of bbox size)
        self.expand_horizontal_range = params.get('expand_horizontal_range', (0.3, 2.0))
        self.expand_vertical_range = params.get('expand_vertical_range', (0.3, 2.0))
        
        # Intensity of brightness
        self.intensity_range = params.get('intensity_range', (1.4, 2.5))
        
        # Falloff type
        self.decay = params.get('decay', 'gaussian')  # 'gaussian' or 'linear'
        
        # Color bias
        self.color_bias = params.get('color_bias', 'auto')  # 'auto', 'warm', 'cool', 'white'
        
        # Edge-only mode (illuminate only one side)
        self.edge_only_prob = params.get('edge_only_prob', 0.3)
    
    def _get_illumination_color(self, image, bias='auto'):
        """Generate illumination color based on bias."""
        if bias == 'warm':
            return np.array([255, 220, 180], dtype=np.float32)
        elif bias == 'cool':
            return np.array([180, 200, 255], dtype=np.float32)
        elif bias == 'white':
            return np.array([255, 255, 255], dtype=np.float32)
        else:  # auto - use image average with slight warm bias
            avg_color = np.mean(image, axis=(0, 1))
            # Add warm bias
            avg_color[0] = min(255, avg_color[0] * 1.1)  # R
            avg_color[1] = min(255, avg_color[1] * 1.05)  # G
            avg_color[2] = min(255, avg_color[2] * 0.95)  # B
            return avg_color
    
    def apply(self, image, bboxes, labels):
        if not bboxes:
            return image, bboxes, labels
        
        aug_image = image.copy()
        img_h, img_w = image.shape[:2]
        
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            
            # Get bbox coordinates
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Random asymmetric expansion
            expand_left = random.uniform(*self.expand_horizontal_range) * bbox_w
            expand_right = random.uniform(*self.expand_horizontal_range) * bbox_w
            expand_top = random.uniform(*self.expand_vertical_range) * bbox_h
            expand_bottom = random.uniform(*self.expand_vertical_range) * bbox_h
            
            # Create outer rectangle
            rect_x1 = max(0, int(x1 - expand_left))
            rect_y1 = max(0, int(y1 - expand_top))
            rect_x2 = min(img_w, int(x2 + expand_right))
            rect_y2 = min(img_h, int(y2 + expand_bottom))
            
            # Decide if edge-only illumination
            edge_only = random.random() < self.edge_only_prob
            
            # Import the new function
            from .utils import rectangular_gradient_mask
            
            # Create rectangular gradient mask
            mask = rectangular_gradient_mask(
                img_h, img_w, x1, y1, x2, y2,
                rect_x1, rect_y1, rect_x2, rect_y2,
                falloff=self.decay,
                edge_only=edge_only
            )
            
            # Get illumination intensity
            intensity = random.uniform(*self.intensity_range)
            
            # Get illumination color
            illum_color = self._get_illumination_color(aug_image, self.color_bias)
            
            # Create bright overlay
            bright = np.clip(
                aug_image.astype(np.float32) * intensity * 0.7 + 
                illum_color * 0.3,
                0, 255
            ).astype(np.uint8)
            
            # Blend with mask
            from .utils import apply_masked_blend
            aug_image = apply_masked_blend(aug_image, bright, mask, alpha=1.0)
        
        return aug_image, bboxes, labels


# ============================================================
# ConcentratedNoiseTransform
# ============================================================
class ConcentratedNoiseTransform(BaseTransform):
    """Intense Gaussian noise near bbox center, fading outward."""
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.sigma_center = params.get('sigma_center', (25, 60))
        self.sigma_background = params.get('sigma_background', (3, 15))

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        base_noise = np.random.randn(img_h, img_w, 3) * random.uniform(*self.sigma_background)
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * 1.5)
            mask = radial_mask(img_h, img_w, cx, cy, radius)
            local_noise = np.random.randn(img_h, img_w, 3) * random.uniform(*self.sigma_center)
            base_noise = base_noise * (1 - mask[..., None]) + local_noise * mask[..., None]
        out = np.clip(image.astype(np.float32) + base_noise, 0, 255).astype(np.uint8)
        return out, bboxes, labels


# ============================================================
# BallBlendAndShapeBiasTransform
# ============================================================
class BallBlendAndShapeBiasTransform(BaseTransform):
    """Shape warp and color blending with background."""
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.warp_strength = params.get('warp_strength', (0.05, 0.25))

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        bg_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            fx = 1 + random.uniform(*self.warp_strength) * random.choice([-1, 1])
            fy = 1 + random.uniform(*self.warp_strength) * random.choice([-1, 1])
            # compute intermediate resize safely
            new_w = max(1, int(round(crop.shape[1] * fx)))
            new_h = max(1, int(round(crop.shape[0] * fy)))
            warped = cv2.resize(crop, (new_w, new_h))
            warped = cv2.resize(warped, (x2 - x1, y2 - y1))
            blend = cv2.addWeighted(warped, 0.7, np.full_like(warped, bg_color, dtype=np.uint8), 0.3, 0)
            image[y1:y2, x1:x2] = blend
        return image, bboxes, labels


# ============================================================
# BallPixelLevelOcclusion
# ============================================================
class BallPixelLevelOcclusion(BaseTransform):
    """Random pixel dropout and micro-occlusions."""
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.dropout_frac = params.get('dropout_frac', (0.02, 0.25))

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frac = random.uniform(*self.dropout_frac)
            mask = np.random.rand(*crop.shape[:2]) < frac
            # replace masked pixels with random colors
            num = np.sum(mask)
            if num > 0:
                crop[mask] = np.random.randint(0, 256, (int(num), 3), dtype=np.uint8)
            image[y1:y2, x1:x2] = crop
        return image, bboxes, labels


# ============================================================
# GradientPatchTransform
# ============================================================
class GradientPatchTransform(BaseTransform):
    """
    Smooth rectangular gradient illumination around bbox.
    Simulates directional light sources, reflections, or shadows.
    """
    
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        
        # Patch type determines expansion pattern
        self.patch_type = params.get('patch_type', 'radial')  # kept for compatibility
        
        # Rectangular expansion
        self.expand_horizontal_range = params.get('expand_horizontal_range', (0.5, 1.8))
        self.expand_vertical_range = params.get('expand_vertical_range', (0.5, 1.8))
        
        # Intensity range
        self.intensity_range = params.get('intensity_range', (0.6, 2.0))
        
        # Color shift
        self.color_shift_scale = params.get('color_shift_scale', 0.3)
        
        # Blend profile
        self.blend_profile = params.get('blend_profile', 'gaussian')
        
        # Directional probability (one side vs all sides)
        self.directional_prob = params.get('directional_prob', 0.4)
    
    def apply(self, image, bboxes, labels):
        if not bboxes:
            return image, bboxes, labels
        
        aug_image = image.copy()
        img_h, img_w = image.shape[:2]
        
        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue
            
            # Get bbox coordinates
            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Random expansion
            expand_h = random.uniform(*self.expand_horizontal_range)
            expand_v = random.uniform(*self.expand_vertical_range)
            
            expand_left = expand_h * bbox_w
            expand_right = expand_h * bbox_w
            expand_top = expand_v * bbox_h
            expand_bottom = expand_v * bbox_h
            
            # Create outer rectangle
            rect_x1 = max(0, int(x1 - expand_left))
            rect_y1 = max(0, int(y1 - expand_top))
            rect_x2 = min(img_w, int(x2 + expand_right))
            rect_y2 = min(img_h, int(y2 + expand_bottom))
            
            # Directional lighting
            edge_only = random.random() < self.directional_prob
            
            # Import the new function
            from .utils import rectangular_gradient_mask, apply_masked_blend
            
            # Create mask
            mask = rectangular_gradient_mask(
                img_h, img_w, x1, y1, x2, y2,
                rect_x1, rect_y1, rect_x2, rect_y2,
                falloff=self.blend_profile,
                edge_only=edge_only
            )
            
            # Intensity
            intensity = random.uniform(*self.intensity_range)
            
            # Color shift
            color_shift = np.random.uniform(-1, 1, 3) * self.color_shift_scale * 50
            
            # Create gradient overlay
            overlay = np.clip(
                aug_image.astype(np.float32) * intensity + color_shift,
                0, 255
            ).astype(np.uint8)
            
            # Blend
            aug_image = apply_masked_blend(aug_image, overlay, mask, alpha=0.8)
        
        return aug_image, bboxes, labels


class BboxGaussianOccludeShearTransform(BaseTransform):
    """
    Applies Gaussian-shaped occlusion masks over bbox and optional shear.
    - Simulates fog, partial transparency, or light reflections under motion.
    """
    def __init__(self, **params):
        bbox_prob = params.get('bbox_prob', 1.0)
        super().__init__(bbox_prob=bbox_prob)
        self.num_patches = params.get('num_patches', (1, 3))
        self.intensity_range = params.get('intensity_range', (0.5, 1.5))
        self.shear_x_range = params.get('shear_x_range', (-50, 50))
        self.shear_y_range = params.get('shear_y_range', (-20, 20))
        self.color_mode = params.get('color_mode', 'background')  # 'background' or 'ball'
        self.blur_after = params.get('blur_after', True)

    def apply(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]
        base_copy = image.copy()

        for box in bboxes:
            if random.random() > self.bbox_prob:
                continue

            x1, y1, x2, y2 = self.yolo_to_pixels(box, (img_h, img_w))
            x1, y1, x2, y2 = self.clip_bbox_to_image(x1, y1, x2, y2, (img_h, img_w))
            if x2 <= x1 or y2 <= y1:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # ---- Step 1: Create Gaussian occlusion mask ----
            # ensure valid randint bounds
            num_min = int(self.num_patches[0]) if isinstance(self.num_patches, (list, tuple)) else int(self.num_patches)
            num_max = int(self.num_patches[1]) if isinstance(self.num_patches, (list, tuple)) else num_min
            if num_max < num_min:
                num_min, num_max = num_max, num_min
            patch_count = random.randint(max(1, num_min), max(1, num_max))

            mask_total = np.zeros(crop.shape[:2], np.float32)

            for _ in range(patch_count):
                # random patch center within bbox (safe bounds)
                px = random.randint(0, max(0, crop.shape[1] - 1))
                py = random.randint(0, max(0, crop.shape[0] - 1))
                min_r = max(1, int(0.2 * (x2 - x1)))
                max_r = max(min_r, int(0.8 * (x2 - x1)))
                radius = random.randint(min_r, max_r)
                mask = radial_mask(crop.shape[0], crop.shape[1], px, py, radius)
                intensity = random.uniform(*self.intensity_range)
                mask_total += mask * intensity

            mask_total = np.clip(mask_total, 0, 1)

            # ---- Step 2: Choose occlusion color ----
            if self.color_mode == 'background':
                color = np.mean(base_copy, axis=(0, 1))
            elif self.color_mode == 'ball':
                color = np.mean(crop, axis=(0, 1)) if crop.size else np.array([255, 255, 255])
            else:
                color = np.array([255, 255, 255])  # white fog by default

            occlusion_layer = np.full_like(crop, color, dtype=np.uint8)
            crop = apply_masked_blend(crop, occlusion_layer, mask_total, alpha=1.0)

            # ---- Step 3: Optional Shear ----
            if random.random() < 0.9:  # 90% of time
                sx, sy = random.uniform(*self.shear_x_range), random.uniform(*self.shear_y_range)
                M = np.float32([[1, np.tan(np.radians(sx)) / 50, 0],
                                [np.tan(np.radians(sy)) / 50, 1, 0]])
                crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                      borderMode=cv2.BORDER_REFLECT)

            # ---- Step 4: Optional Blur ----
            if self.blur_after and random.random() < 0.7:
                k = random.choice([3, 5, 7])
                crop = cv2.GaussianBlur(crop, (k, k), 0)

            # ---- Step 5: Write back ----
            image[y1:y2, x1:x2] = crop

        return image, bboxes, labels