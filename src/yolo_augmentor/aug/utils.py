# utils_augment.py
# Shared utilities for localized effects and smooth blending

import numpy as np
import cv2
import random as rand

def radial_mask(h, w, center_x, center_y, radius, falloff='gaussian'):
    """
    Creates a smooth radial mask that decays from center to edges.
    - falloff: 'gaussian' (smooth) or 'linear' (harder)
    """
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - center_x)**2 + (Y - center_y)**2
    if falloff == 'gaussian':
        sigma = radius / 2.0
        mask = np.exp(-dist2 / (2 * sigma * sigma))
    else:
        dist = np.sqrt(dist2)
        mask = np.clip(1 - dist / radius, 0.0, 1.0)
    return mask.astype(np.float32)

def rectangular_gradient_mask(h, w, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                              outer_x1, outer_y1, outer_x2, outer_y2,
                              falloff='gaussian', edge_only=False):
    """
    Creates a rectangular gradient mask around the bbox.
    Illumination is OUTSIDE the bbox, fading toward outer edges.
    
    Args:
        h, w: Image dimensions
        bbox_x1, bbox_y1, bbox_x2, bbox_y2: Inner bbox coordinates
        outer_x1, outer_y1, outer_x2, outer_y2: Outer rectangle coordinates
        falloff: 'gaussian' or 'linear'
        edge_only: If True, only illuminate one random edge
    
    Returns:
        mask: Float32 array [h, w] with values in [0, 1]
    """
    mask = np.zeros((h, w), dtype=np.float32)
    Y, X = np.ogrid[:h, :w]
    
    # Calculate distances to bbox edges (positive = outside bbox)
    dist_left = X - bbox_x1
    dist_right = bbox_x2 - X
    dist_top = Y - bbox_y1
    dist_bottom = bbox_y2 - Y
    
    # Check if point is outside bbox
    outside_left = X < bbox_x1
    outside_right = X >= bbox_x2
    outside_top = Y < bbox_y1
    outside_bottom = Y >= bbox_y2
    
    # Check if point is inside outer rectangle
    in_outer = (X >= outer_x1) & (X < outer_x2) & (Y >= outer_y1) & (Y < outer_y2)
    
    if edge_only:
        # Randomly select one edge to illuminate
        edge = rand.choice(['left', 'right', 'top', 'bottom'])
        
        if edge == 'left':
            region = outside_left & in_outer & (Y >= bbox_y1) & (Y < bbox_y2)
            distance = bbox_x1 - X
        elif edge == 'right':
            region = outside_right & in_outer & (Y >= bbox_y1) & (Y < bbox_y2)
            distance = X - bbox_x2
        elif edge == 'top':
            region = outside_top & in_outer & (X >= bbox_x1) & (X < bbox_x2)
            distance = bbox_y1 - Y
        else:  # bottom
            region = outside_bottom & in_outer & (X >= bbox_x1) & (X < bbox_x2)
            distance = Y - bbox_y2
        
        max_dist = max(outer_x2 - outer_x1, outer_y2 - outer_y1) * 0.5
        
    else:
        # Illuminate all sides around bbox
        outside_bbox = outside_left | outside_right | outside_top | outside_bottom
        region = outside_bbox & in_outer
        
        # Distance to nearest bbox edge
        distance = np.minimum(
            np.minimum(np.abs(dist_left), np.abs(dist_right)),
            np.minimum(np.abs(dist_top), np.abs(dist_bottom))
        )
        
        max_dist = max(bbox_x2 - bbox_x1, bbox_y2 - bbox_y1)
    
    # Apply falloff
    if falloff == 'gaussian':
        sigma = max_dist * 0.4
        mask = np.where(region, np.exp(-distance**2 / (2 * sigma**2)), 0.0)
    else:  # linear
        mask = np.where(region, np.clip(1.0 - distance / max_dist, 0.0, 1.0), 0.0)
    
    return mask.astype(np.float32)


def apply_masked_blend(base, overlay, mask, alpha=1.0):
    """
    Blends overlay into base image using mask ∈ [0,1].
    """
    return np.clip(
        base.astype(np.float32) * (1 - mask[..., None]) +
        overlay.astype(np.float32) * mask[..., None] * alpha,
        0, 255
    ).astype(np.uint8)

def random_occlusion_mask(h, w, intensity='medium'):
    """
    Generates a random rectangular occlusion mask.
    """
    mask = np.zeros((h, w), np.uint8)
    n_rects = {'low': 1, 'medium': 2, 'high': 4}.get(intensity, 2)
    for _ in range(n_rects):
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = np.random.randint(x1 + 5, w)
        y2 = np.random.randint(y1 + 5, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask