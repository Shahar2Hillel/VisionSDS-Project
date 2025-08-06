# Fixed BlenderProc script with proper multi-tool integration
import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse
import random
import os
import json
import glob
from colorsys import hsv_to_rgb
from blenderproc.python.camera.CameraProjection import project_points
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  
from PIL import Image
from tqdm import tqdm

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

def enhance_background_integration(rendered_image, backgrounds_dir):
    """Paste background and simulate surgical light effect"""
    background_files = get_background_files(backgrounds_dir)
    if not background_files:
        return rendered_image

    # Convert to uint8 image
    if rendered_image.dtype != np.uint8:
        rendered_image = (rendered_image * 255).astype(np.uint8)

    if rendered_image.shape[2] == 4:
        pil_foreground = Image.fromarray(rendered_image, 'RGBA')
    else:
        rgb = rendered_image
        alpha = np.where(np.sum(rgb, axis=2) > 0.01, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        pil_foreground = Image.fromarray(rgba, 'RGBA')

    width, height = pil_foreground.size

    # Load background image
    bg_path = random.choice(background_files)
    background = Image.open(bg_path).convert('RGB').resize((width, height))

    # Apply overall dimming
    lighting_factor = random.uniform(0.6, 0.9)  # Simulate ambient OR light
    background = ImageEnhance.Brightness(background).enhance(lighting_factor)

    # Simulate surgical spotlight (radial brightening)
    spotlight = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(spotlight)

    # Center of spotlight: assume center of non-transparent pixels
    alpha = pil_foreground.split()[-1]
    alpha_np = np.array(alpha)
    nonzero = np.argwhere(alpha_np > 10)

    if len(nonzero) > 0:
        center_y, center_x = np.mean(nonzero, axis=0).astype(int)
    else:
        center_x, center_y = width // 2, height // 2

    # Draw radial gradient for spotlight
    max_radius = int(min(width, height) * random.uniform(0.3, 0.5))
    for r in range(max_radius, 0, -1):
        intensity = int(255 * (1 - r / max_radius) ** 2)
        bbox = [center_x - r, center_y - r, center_x + r, center_y + r]
        draw.ellipse(bbox, fill=intensity)

    # Blur the spotlight mask for softness
    spotlight = spotlight.filter(ImageFilter.GaussianBlur(radius=30))

    # Convert spotlight to RGB and blend with background
    spotlight_rgb = Image.merge('RGB', [spotlight] * 3)
    spotlight_blended = Image.blend(background, spotlight_rgb, alpha=0.4)

    # Composite final image
    result = Image.new('RGBA', (width, height))
    result.paste(spotlight_blended, (0, 0))
    result.paste(pil_foreground, (0, 0), mask=pil_foreground)

    final_image = Image.new('RGB', (width, height), (255, 255, 255))
    final_image.paste(result, mask=result.split()[-1])

    return np.array(final_image) / 255.0

class KeypointDetector:
    """Class to handle category-specific keypoint detection"""
    
    @staticmethod
    def detect_needle_holder_keypoints(verts_world):
        """
        Detect keypoints specific to needle holders:
        - top_left: Left jaw tip (highest Z, left side)
        - top_right: Right jaw tip (highest Z, right side)  
        - bottom_left: Bottom left corner
        - bottom_right: Bottom right corner
        - middle_left: Midpoint along the left jaw
        - middle_right: Midpoint along the right jaw
        - joint_center: Geometrically calculated joint between jaws and shaft
        """
        verts_array = np.array(verts_world)
        x_coords = verts_array[:, 0]
        y_coords = verts_array[:, 1]
        z_coords = verts_array[:, 2]
        
        keypoints = {}
        
        # 1. Detect jaw tips (top_left, top_right) - these should be at the actual tips
        # Find vertices in the top 15% of Z coordinates (actual tips)
        top_z_threshold = np.percentile(z_coords, 85)
        tip_candidates_indices = np.where(z_coords >= top_z_threshold)[0]
        
        if len(tip_candidates_indices) > 0:
            tip_candidates = verts_array[tip_candidates_indices]
            tip_x_coords = tip_candidates[:, 0]
            
            # Left tip: leftmost point among tip candidates
            left_tip_idx = tip_candidates_indices[np.argmin(tip_x_coords)]
            keypoints['top_left'] = verts_array[left_tip_idx]
            
            # Right tip: rightmost point among tip candidates  
            right_tip_idx = tip_candidates_indices[np.argmax(tip_x_coords)]
            keypoints['top_right'] = verts_array[right_tip_idx]
        else:
            # Fallback: use absolute extremes
            z_max_idx = np.argmax(z_coords)
            z_max_point = verts_array[z_max_idx]
            keypoints['top_left'] = z_max_point
            keypoints['top_right'] = z_max_point
        
        # 2. Bottom corners (keep existing logic)
        bottom_z_threshold = np.percentile(z_coords, 15)
        bottom_candidates_indices = np.where(z_coords <= bottom_z_threshold)[0]
        
        if len(bottom_candidates_indices) > 0:
            bottom_candidates = verts_array[bottom_candidates_indices]
            bottom_x_coords = bottom_candidates[:, 0]
            
            # Bottom left: leftmost point among bottom candidates
            bottom_left_idx = bottom_candidates_indices[np.argmin(bottom_x_coords)]
            keypoints['bottom_left'] = verts_array[bottom_left_idx]
            
            # Bottom right: rightmost point among bottom candidates
            bottom_right_idx = bottom_candidates_indices[np.argmax(bottom_x_coords)]
            keypoints['bottom_right'] = verts_array[bottom_right_idx]
        else:
            # Fallback
            z_min_idx = np.argmin(z_coords)
            z_min_point = verts_array[z_min_idx]
            keypoints['bottom_left'] = z_min_point
            keypoints['bottom_right'] = z_min_point
        
        # 3. Middle jaw points (keep existing logic)
        mid_z_min = np.percentile(z_coords, 30)
        mid_z_max = np.percentile(z_coords, 70)
        jaw_mask = (z_coords >= mid_z_min) & (z_coords <= mid_z_max)
        jaw_candidates_indices = np.where(jaw_mask)[0]
        
        if len(jaw_candidates_indices) > 0:
            jaw_candidates = verts_array[jaw_candidates_indices]
            jaw_x_coords = jaw_candidates[:, 0]
            
            # Left jaw (minimum X in jaw region)
            left_jaw_mask = jaw_x_coords <= np.percentile(jaw_x_coords, 25)
            left_jaw_indices = jaw_candidates_indices[left_jaw_mask]
            if len(left_jaw_indices) > 0:
                left_jaw_points = verts_array[left_jaw_indices]
                left_centroid = np.mean(left_jaw_points, axis=0)
                distances = np.linalg.norm(left_jaw_points - left_centroid, axis=1)
                closest_idx = left_jaw_indices[np.argmin(distances)]
                keypoints['middle_left'] = verts_array[closest_idx]
            
            # Right jaw (maximum X in jaw region)
            right_jaw_mask = jaw_x_coords >= np.percentile(jaw_x_coords, 75)
            right_jaw_indices = jaw_candidates_indices[right_jaw_mask]
            if len(right_jaw_indices) > 0:
                right_jaw_points = verts_array[right_jaw_indices]
                right_centroid = np.mean(right_jaw_points, axis=0)
                distances = np.linalg.norm(right_jaw_points - right_centroid, axis=1)
                closest_idx = right_jaw_indices[np.argmin(distances)]
                keypoints['middle_right'] = verts_array[right_jaw_indices[np.argmin(distances)]]
        
        # Fallback for middle points if jaw detection fails
        if 'middle_left' not in keypoints or 'middle_right' not in keypoints:
            median_x = np.median(x_coords)
            left_indices = np.where(x_coords <= median_x)[0]
            right_indices = np.where(x_coords >= median_x)[0]
            
            if len(left_indices) > 0:
                left_points = verts_array[left_indices]
                left_centroid = np.mean(left_points, axis=0)
                distances = np.linalg.norm(left_points - left_centroid, axis=1)
                keypoints['middle_left'] = left_points[np.argmin(distances)]
            
            if len(right_indices) > 0:
                right_points = verts_array[right_indices]
                right_centroid = np.mean(right_points, axis=0)
                distances = np.linalg.norm(right_points - right_centroid, axis=1)
                keypoints['middle_right'] = right_points[np.argmin(distances)]
        
        # 4. FIXED: Geometric joint center calculation
        # Calculate joint as a point between the tips and the shaft midpoints
        if 'top_left' in keypoints and 'top_right' in keypoints and 'middle_left' in keypoints and 'middle_right' in keypoints:
            # Get the tip points and middle points
            left_tip = keypoints['top_left']
            right_tip = keypoints['top_right']
            left_middle = keypoints['middle_left']
            right_middle = keypoints['middle_right']
            
            # Calculate the center between tips
            tips_center = (left_tip + right_tip) / 2.0
            
            # Calculate the center between middle points (shaft center)
            shaft_center = (left_middle + right_middle) / 2.0
            
            # Joint is positioned 30% of the way from tips toward shaft
            # This creates a geometrically meaningful joint position
            joint_offset_ratio = 0.3  # 30% toward shaft from tips
            joint_position = tips_center + joint_offset_ratio * (shaft_center - tips_center)
            
            # Find the actual vertex closest to this calculated joint position
            distances_to_joint = np.linalg.norm(verts_array - joint_position, axis=1)
            closest_joint_idx = np.argmin(distances_to_joint)
            keypoints['joint_center'] = verts_array[closest_joint_idx]
            
        else:
            # Fallback: use the existing statistical method
            joint_z_max = np.percentile(z_coords, 40)  # Lower 40% of Z values
            joint_candidates_indices = np.where(z_coords <= joint_z_max)[0]
            
            if len(joint_candidates_indices) > 0:
                joint_candidates = verts_array[joint_candidates_indices]
                
                # Find the point closest to the X-center among joint candidates
                x_center = np.median(x_coords)
                x_distances = np.abs(joint_candidates[:, 0] - x_center)
                
                # Among X-centered candidates, find the one with median Y coordinate
                center_threshold = np.percentile(x_distances, 30)  # Take X-centered points
                x_centered_mask = x_distances <= center_threshold
                
                if np.any(x_centered_mask):
                    x_centered_candidates = joint_candidates[x_centered_mask]
                    x_centered_indices = joint_candidates_indices[x_centered_mask]
                    
                    # Among X-centered points, find the one closest to median Y
                    y_center = np.median(x_centered_candidates[:, 1])
                    y_distances = np.abs(x_centered_candidates[:, 1] - y_center)
                    best_joint_local_idx = np.argmin(y_distances)
                    best_joint_idx = x_centered_indices[best_joint_local_idx]
                    
                    keypoints['joint_center'] = verts_array[best_joint_idx]
                else:
                    # Fallback: use the point closest to center
                    center_distances = np.sqrt(x_distances**2 + 
                                             (joint_candidates[:, 1] - np.median(joint_candidates[:, 1]))**2)
                    best_joint_local_idx = np.argmin(center_distances)
                    best_joint_idx = joint_candidates_indices[best_joint_local_idx]
                    keypoints['joint_center'] = verts_array[best_joint_idx]
            else:
                # Final fallback: use bottom center point
                z_min_indices = np.where(z_coords <= np.percentile(z_coords, 20))[0]
                if len(z_min_indices) > 0:
                    bottom_points = verts_array[z_min_indices]
                    x_center = np.median(x_coords)
                    x_distances = np.abs(bottom_points[:, 0] - x_center)
                    closest_to_center_local = np.argmin(x_distances)
                    keypoints['joint_center'] = bottom_points[closest_to_center_local]
                else:
                    keypoints['joint_center'] = verts_array[np.argmin(z_coords)]
        
        return keypoints
        
    @staticmethod
    def detect_tweezers_keypoints(verts_world):
        """
        Detect keypoints specific to tweezers:
        - bottom_tip: The lowest point where tweezers pinch
        - mid_left: Midpoint along the left arm
        - mid_right: Midpoint along the right arm  
        - top_left: Furthest back/top point on the left arm (FIXED: actual tip vertex)
        - top_right: Furthest back/top point on the right arm (FIXED: actual tip vertex)
        """
        verts_array = np.array(verts_world)
        x_coords = verts_array[:, 0]
        y_coords = verts_array[:, 1]
        z_coords = verts_array[:, 2]
        
        keypoints = {}
        
        # Bottom tip: absolute minimum Z point (pinching end) - actual vertex
        min_z_idx = np.argmin(z_coords)
        keypoints['bottom_tip'] = verts_array[min_z_idx]
        
        # Split vertices into left and right arms based on X coordinate
        median_x = np.median(x_coords)
        left_arm_indices = np.where(x_coords <= median_x)[0]
        right_arm_indices = np.where(x_coords >= median_x)[0]
        
        # Process left arm
        if len(left_arm_indices) > 0:
            left_arm_verts = verts_array[left_arm_indices]
            left_z_coords = left_arm_verts[:, 2]
            left_x_coords = left_arm_verts[:, 0]
            
            # FIXED: Top left - find the actual tip vertex (high Z AND extreme left X)
            # First filter to top 20% of Z coordinates in left arm
            top_z_threshold_left = np.percentile(left_z_coords, 80)
            top_candidates_mask = left_z_coords >= top_z_threshold_left
            
            if np.any(top_candidates_mask):
                # Among the high-Z candidates, find the leftmost one (most extreme X)
                top_candidates_x = left_x_coords[top_candidates_mask]
                leftmost_among_tops_idx = np.argmin(top_candidates_x)
                
                # Convert back to original indices
                top_candidates_indices = np.where(top_candidates_mask)[0]
                top_left_local_idx = top_candidates_indices[leftmost_among_tops_idx]
                keypoints['top_left'] = left_arm_verts[top_left_local_idx]
            else:
                # Fallback: use the highest Z in left arm
                top_left_local_idx = np.argmax(left_z_coords)
                keypoints['top_left'] = left_arm_verts[top_left_local_idx]
            
            # Mid left: vertex closest to median Z point in left arm - actual vertex
            median_left_z = np.median(left_z_coords)
            z_distances = np.abs(left_z_coords - median_left_z)
            mid_left_local_idx = np.argmin(z_distances)
            keypoints['mid_left'] = left_arm_verts[mid_left_local_idx]
        
        # Process right arm
        if len(right_arm_indices) > 0:
            right_arm_verts = verts_array[right_arm_indices]
            right_z_coords = right_arm_verts[:, 2]
            right_x_coords = right_arm_verts[:, 0]
            
            # FIXED: Top right - find the actual tip vertex (high Z AND extreme right X)
            # First filter to top 20% of Z coordinates in right arm
            top_z_threshold_right = np.percentile(right_z_coords, 80)
            top_candidates_mask = right_z_coords >= top_z_threshold_right
            
            if np.any(top_candidates_mask):
                # Among the high-Z candidates, find the rightmost one (most extreme X)
                top_candidates_x = right_x_coords[top_candidates_mask]
                rightmost_among_tops_idx = np.argmax(top_candidates_x)
                
                # Convert back to original indices
                top_candidates_indices = np.where(top_candidates_mask)[0]
                top_right_local_idx = top_candidates_indices[rightmost_among_tops_idx]
                keypoints['top_right'] = right_arm_verts[top_right_local_idx]
            else:
                # Fallback: use the highest Z in right arm
                top_right_local_idx = np.argmax(right_z_coords)
                keypoints['top_right'] = right_arm_verts[top_right_local_idx]
            
            # Mid right: vertex closest to median Z point in right arm - actual vertex
            median_right_z = np.median(right_z_coords)
            z_distances = np.abs(right_z_coords - median_right_z)
            mid_right_local_idx = np.argmin(z_distances)
            keypoints['mid_right'] = right_arm_verts[mid_right_local_idx]
        
        return keypoints
    
    @staticmethod
    def _get_standard_corners_actual_verts(verts_array, x_coords, y_coords, z_coords):
        """Get standard corner keypoints using actual vertex positions"""
        def find_corner_vertex(x_coords, z_coords, z_preference, x_preference, verts_array):
            if z_preference == 'min':
                z_candidates = np.where(z_coords <= np.percentile(z_coords, 20))[0]
            else:
                z_candidates = np.where(z_coords >= np.percentile(z_coords, 80))[0]
            
            if len(z_candidates) > 0:
                z_candidate_x = x_coords[z_candidates]
                if x_preference == 'min':
                    relative_idx = np.argmin(z_candidate_x)
                else:
                    relative_idx = np.argmax(z_candidate_x)
                return verts_array[z_candidates[relative_idx]]
            else:
                # Fallback to absolute extremes
                if z_preference == 'min' and x_preference == 'min':
                    # Find vertex that minimizes z+x (bottom-left tendency)
                    scores = z_coords + x_coords
                    return verts_array[np.argmin(scores)]
                elif z_preference == 'min' and x_preference == 'max':
                    # Find vertex that minimizes z-x (bottom-right tendency)
                    scores = z_coords - x_coords
                    return verts_array[np.argmin(scores)]
                elif z_preference == 'max' and x_preference == 'min':
                    # Find vertex that maximizes z-x (top-left tendency)
                    scores = z_coords - x_coords
                    return verts_array[np.argmax(scores)]
                else:
                    # Find vertex that maximizes z+x (top-right tendency)
                    scores = z_coords + x_coords
                    return verts_array[np.argmax(scores)]
        
        return {
            'bottom_left': find_corner_vertex(x_coords, z_coords, 'min', 'min', verts_array),
            'bottom_right': find_corner_vertex(x_coords, z_coords, 'min', 'max', verts_array),
            'top_left': find_corner_vertex(x_coords, z_coords, 'max', 'min', verts_array),
            'top_right': find_corner_vertex(x_coords, z_coords, 'max', 'max', verts_array)
        }
    
    @staticmethod
    def assign_view_dependent_labels(keypoints_3d, keypoints_2d_projected, frame_idx):
        """
        Reassign left/right labels based on actual 2D positions in the image.
        This fixes the flipping issue when tools are viewed from different angles.
        """
        if not keypoints_2d_projected or len(keypoints_2d_projected) == 0:
            return keypoints_2d_projected
        
        # Create a mapping from keypoint names to their 2D positions
        kp_2d_dict = {}
        kp_names = list(keypoints_3d.keys())
        
        for i, kp_name in enumerate(kp_names):
            if i < len(keypoints_2d_projected) and keypoints_2d_projected[i] is not None:
                kp_2d_dict[kp_name] = keypoints_2d_projected[i]
        
        # If we don't have enough keypoints, return as-is
        if len(kp_2d_dict) < 2:
            return {kp_name: ([float(pt[0]), float(pt[1])] if pt is not None else None) 
                    for kp_name, pt in zip(kp_names, keypoints_2d_projected)}
        
        # Find pairs that need left/right reassignment
        reassigned = {}
        
        # Handle standard corners (bottom_left/bottom_right, top_left/top_right)
        if 'bottom_left' in kp_2d_dict and 'bottom_right' in kp_2d_dict:
            bl_pt = kp_2d_dict['bottom_left']
            br_pt = kp_2d_dict['bottom_right']
            
            # Reassign based on actual X positions in 2D
            if bl_pt[0] <= br_pt[0]:  # Correct assignment
                reassigned['bottom_left'] = [float(bl_pt[0]), float(bl_pt[1])]
                reassigned['bottom_right'] = [float(br_pt[0]), float(br_pt[1])]
            else:  # Flipped - swap them
                reassigned['bottom_left'] = [float(br_pt[0]), float(br_pt[1])]
                reassigned['bottom_right'] = [float(bl_pt[0]), float(bl_pt[1])]
        
        if 'top_left' in kp_2d_dict and 'top_right' in kp_2d_dict:
            tl_pt = kp_2d_dict['top_left']
            tr_pt = kp_2d_dict['top_right']
            
            # Reassign based on actual X positions in 2D
            if tl_pt[0] <= tr_pt[0]:  # Correct assignment
                reassigned['top_left'] = [float(tl_pt[0]), float(tl_pt[1])]
                reassigned['top_right'] = [float(tr_pt[0]), float(tr_pt[1])]
            else:  # Flipped - swap them
                reassigned['top_left'] = [float(tr_pt[0]), float(tr_pt[1])]
                reassigned['top_right'] = [float(tl_pt[0]), float(tl_pt[1])]
        
        # Handle middle points (middle_left/middle_right)
        if 'middle_left' in kp_2d_dict and 'middle_right' in kp_2d_dict:
            ml_pt = kp_2d_dict['middle_left']
            mr_pt = kp_2d_dict['middle_right']
            
            if ml_pt[0] <= mr_pt[0]:  # Correct assignment
                reassigned['middle_left'] = [float(ml_pt[0]), float(ml_pt[1])]
                reassigned['middle_right'] = [float(mr_pt[0]), float(mr_pt[1])]
            else:  # Flipped - swap them
                reassigned['middle_left'] = [float(mr_pt[0]), float(mr_pt[1])]
                reassigned['middle_right'] = [float(ml_pt[0]), float(ml_pt[1])]
        
        # Handle tweezers-specific points (mid_left/mid_right)
        if 'mid_left' in kp_2d_dict and 'mid_right' in kp_2d_dict:
            ml_pt = kp_2d_dict['mid_left']
            mr_pt = kp_2d_dict['mid_right']
            
            if ml_pt[0] <= mr_pt[0]:  # Correct assignment
                reassigned['mid_left'] = [float(ml_pt[0]), float(ml_pt[1])]
                reassigned['mid_right'] = [float(mr_pt[0]), float(mr_pt[1])]
            else:  # Flipped - swap them
                reassigned['mid_left'] = [float(mr_pt[0]), float(mr_pt[1])]
                reassigned['mid_right'] = [float(ml_pt[0]), float(ml_pt[1])]
        
        # Add any remaining keypoints that don't have left/right pairs
        for kp_name, pt in kp_2d_dict.items():
            if kp_name not in reassigned:
                reassigned[kp_name] = [float(pt[0]), float(pt[1])]
        
        return reassigned
    
    @staticmethod
    def get_keypoint_colors():
        """Define colors for each keypoint type"""
        return {
            # Standard corners
            'bottom_left': 'red',
            'bottom_right': 'blue', 
            'top_left': 'green',
            'top_right': 'orange',
            
            # Needle holder specific
            'joint_center': 'purple',
            'middle_left': 'cyan',
            'middle_right': 'magenta',
            
            # Tweezers specific
            'bottom_tip': 'yellow',
            'mid_left': 'pink',
            'mid_right': 'brown',
            # top_left and top_right use standard colors
        }

def get_hdr_img_paths_from_haven(data_path: str) -> list:
    """Returns .hdr file paths from the given directory."""
    if os.path.exists(data_path):
        data_path = os.path.join(data_path, "hdris")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The folder: {data_path} does not contain a folder name hdris. "
                                    f"Please use the download script.")
    else:
        raise FileNotFoundError(f"The data path does not exists: {data_path}")

    hdr_files = glob.glob(os.path.join(data_path, "*", "*.hdr"))
    hdr_files.sort()  # Ensure deterministic order
    return hdr_files

def get_background_files(backgrounds_dir: str) -> list:
    """Get all background image files from directory."""
    background_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        background_files.extend(glob.glob(os.path.join(backgrounds_dir, f"*.{ext}")))
        background_files.extend(glob.glob(os.path.join(backgrounds_dir, f"*.{ext.upper()}")))
    return background_files

def paste_background_on_image(rendered_image, backgrounds_dir):
    """Paste a random background on a transparent rendered image."""
    background_files = get_background_files(backgrounds_dir)
    if not background_files:
        print(f"Warning: No background files found in {backgrounds_dir}")
        return rendered_image
    
    # Convert to PIL Image first
    if rendered_image.dtype != np.uint8:
        rendered_image = (rendered_image * 255).astype(np.uint8)
    
    # Handle the case where rendered_image might have 4 channels (RGBA) or 3 channels (RGB)
    if rendered_image.shape[2] == 4:
        # RGBA image - use the existing alpha channel from BlenderProc
        pil_image = Image.fromarray(rendered_image, 'RGBA')
    else:
        # RGB image - create alpha channel from non-zero pixels
        rgb = rendered_image
        # Create alpha where non-zero pixels are opaque (more permissive threshold)
        alpha = np.where(np.sum(rgb, axis=2) > 0.01, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        pil_image = Image.fromarray(rgba, 'RGBA')
    
    img_w, img_h = pil_image.size
    
    # Select and load random background
    background_path = random.choice(background_files)
    background = Image.open(background_path).convert('RGB').resize([img_w, img_h])
    
    # Apply background pasting using the reference method
    result = Image.new('RGBA', (img_w, img_h))
    result.paste(background, (0, 0))
    result.paste(pil_image, (0, 0), mask=pil_image)
    
    # Convert to RGB for final output
    final_image = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    final_image.paste(result, mask=result.split()[-1])
    
    # Convert back to numpy array
    return np.array(final_image) / 255.0

def safe_set_principled_shader_value(material, input_name, value):
    """Safely set principled shader value, handling missing inputs gracefully."""
    try:
        material.set_principled_shader_value(input_name, value)
        return True
    except KeyError:
        # Input doesn't exist in this version of Blender
        print(f"Warning: Shader input '{input_name}' not available, skipping...")
        return False

def create_visualization(image, keypoints_2d, keypoint_colors, obj_identifier, category, frame_idx, title_suffix="", bbox=None):
    """Create visualization with keypoints and optional bounding box overlaid on image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    legend_patches = []
    
    # Plot keypoints with colors and legend
    for kp_name, kp_color in keypoint_colors.items():
        if kp_name in keypoints_2d:
            pt = keypoints_2d[kp_name]
            if pt is not None:
                x, y = pt
                ax.plot(x, y, 'o', markersize=6, color=kp_color, markeredgecolor='white', markeredgewidth=1)
                legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))
    
    # Plot bounding box if provided
    if bbox is not None and len(bbox) == 4:
        rect = mpatches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 5, 'bbox', color='lime', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))

    # Add legend
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)
    
    ax.axis('off')
    title = f'{category.replace("_", " ").title()}: {obj_identifier} - Frame {frame_idx}{title_suffix}'
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    
    return fig

def process_multi_objects(obj_paths, camera_params, base_output_dir, num_images=25, 
                         global_images_dir=None, global_viz_dir=None, global_annotations=None, object_counter=None,
                         use_background=False, use_hdri=False, backgrounds_dir=None, haven_path=None,
                         bg_images_dir=None, bg_viz_dir=None, hdri_images_dir=None, hdri_viz_dir=None,
                         bg_annotations=None, hdri_annotations=None):
    """Process multiple objects in a single scene."""
    
    # Determine how many objects to render (1 or 2)
    num_objects = random.choice([1, 2])
    selected_obj_paths = random.sample(obj_paths, min(num_objects, len(obj_paths)))
    
    print(f"\nProcessing {num_objects} objects together:")
    for path in selected_obj_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Load HDR files if needed
    hdr_files = []
    if use_hdri and haven_path:
        try:
            hdr_files = get_hdr_img_paths_from_haven(haven_path)
            if not hdr_files:
                print(f"Warning: No HDR files found in {haven_path}")
                use_hdri = False
        except Exception as e:
            print(f"Warning: Could not load HDR files: {e}")
            use_hdri = False
    
    try:
        # Clear the scene
        bproc.clean_up(clean_up_camera=False)
        
        # Ensure world exists and has proper node setup
        world = bpy.context.scene.world
        if world is None or world.node_tree is None:
            if "World" in bpy.data.worlds:
                world = bpy.data.worlds["World"]
            else:
                world = bpy.data.worlds.new("World")
            
            bpy.context.scene.world = world
            world.use_nodes = True
            
            if not world.node_tree.nodes:
                background_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
                output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
                world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        
        # Load all selected objects
        loaded_objects = []
        all_keypoints_3d = {}
        
        for i, obj_path in enumerate(selected_obj_paths):
            obj_filename = os.path.basename(obj_path)
            obj_name = os.path.splitext(obj_filename)[0]
            
            # Determine category
            parent_folder = os.path.basename(os.path.dirname(obj_path))
            if 'needle_holder' in parent_folder.lower():
                category = 'needle_holder'
                category_id = 1
            elif 'tweezers' in parent_folder.lower():
                category = 'tweezers'
                category_id = 2
            else:
                category = 'unknown'
                category_id = 0
            
            # Generate object identifier
            obj_identifier = f"{obj_name}"
            object_counter['count'] += 1
            
            # Load the object
            obj = bproc.loader.load_obj(obj_path)[0]
            obj.set_cp("category_id", category_id)
            obj.set_cp("obj_identifier", obj_identifier)  # Store identifier for later use
            
            # Position objects side by side if multiple
            if len(selected_obj_paths) > 1:
                offset = [-2.0, 0, 0] if i == 0 else [2.0, 0, 0]
                current_location = obj.get_location()
                obj.set_location(current_location + np.array(offset))
            
            # Apply materials (same as single object processing)
            materials = obj.get_materials()
            if len(materials) > 0:
                mat = materials[0]
                
                # Basic material properties
                safe_set_principled_shader_value(mat, "IOR", random.uniform(1.0, 2.5))
                safe_set_principled_shader_value(mat, "Roughness", random.uniform(0.1, 0.8))
                safe_set_principled_shader_value(mat, "Metallic", 1)
                
                # CRITICAL: Ensure full opacity
                safe_set_principled_shader_value(mat, "Alpha", 1.0)
                safe_set_principled_shader_value(mat, "Transmission", 0.0)
                
                # Set blend mode to opaque to prevent transparency issues
                try:
                    if hasattr(mat.blender_obj, 'blend_method'):
                        mat.blender_obj.blend_method = 'OPAQUE'
                    elif hasattr(mat.blender_obj, 'use_transparency'):
                        mat.blender_obj.use_transparency = False
                except Exception as e:
                    print(f"Warning: Could not set blend mode: {e}")
                
                # Handle second material if present
                if len(materials) > 1:
                    mat2 = materials[1]
                    random_gold_hsv_color = np.random.uniform([0.03, 0.95, 0.8], [0.25, 1.0, 1.0])
                    random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0]
                    
                    safe_set_principled_shader_value(mat2, "Base Color", random_gold_color)
                    safe_set_principled_shader_value(mat2, "IOR", random.uniform(1.0, 2.5))
                    safe_set_principled_shader_value(mat2, "Roughness", random.uniform(0.1, 0.8))
                    safe_set_principled_shader_value(mat2, "Metallic", 1)
                    
                    # CRITICAL: Ensure full opacity
                    safe_set_principled_shader_value(mat2, "Alpha", 1.0)
                    safe_set_principled_shader_value(mat2, "Transmission", 0.0)
                    
                    # Set blend mode to opaque
                    try:
                        if hasattr(mat2.blender_obj, 'blend_method'):
                            mat2.blender_obj.blend_method = 'OPAQUE'
                        elif hasattr(mat2.blender_obj, 'use_transparency'):
                            mat2.blender_obj.use_transparency = False
                    except Exception as e:
                        print(f"Warning: Could not set blend mode for material 2: {e}")
            
            loaded_objects.append({
                'obj': obj,
                'identifier': obj_identifier,
                'category': category,
                'original_name': obj_name
            })
            
            # Get keypoints for this object
            mesh = obj.get_mesh()
            verts = mesh.vertices
            obj2world = obj.get_local2world_mat()
            verts_world = [obj2world @ np.append(v.co, 1.0) for v in verts]
            verts_world = [v[:3] for v in verts_world]
            
            # Detect keypoints based on category - UPDATED TO USE ACTUAL VERTICES
           

            # Detect keypoints based on category
            if category == 'needle_holder':
                keypoints_3d = KeypointDetector.detect_needle_holder_keypoints(verts_world)
                # REMOVED: No longer delete joint_center keypoint - it's now intentionally included
            elif category == 'tweezers':
                keypoints_3d = KeypointDetector.detect_tweezers_keypoints(verts_world)
            else:
                # Fallback to standard corners only
                verts_array = np.array(verts_world)
                keypoints_3d = KeypointDetector._get_standard_corners_actual_verts(
                    verts_array, verts_array[:, 0], verts_array[:, 1], verts_array[:, 2]
                )
            
            all_keypoints_3d[obj_identifier] = keypoints_3d
            print(f"Detected keypoints for {obj_identifier} ({category}):")
            for kp_name, kp_coord in keypoints_3d.items():
                print(f"  {kp_name}: {kp_coord}")
        
        # Create lighting
        light = bproc.types.Light()
        light.set_type("POINT")
        
        # Position light to illuminate all objects
        if len(loaded_objects) > 1:
            # Get center point between all objects
            all_locations = [obj_data['obj'].get_location() for obj_data in loaded_objects]
            scene_center = np.mean(all_locations, axis=0)
        else:
            scene_center = loaded_objects[0]['obj'].get_location()
        
        light.set_location(bproc.sampler.shell(
            center=scene_center,
            radius_min=1,
            radius_max=5,
            elevation_min=1,
            elevation_max=89
        ))
        light.set_energy(random.uniform(100, 1000))
        
        # Set camera intrinsics
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
        im_width = camera_params["width"]
        im_height = camera_params["height"]
        K = np.array([[fx, 0, cx], 
                      [0, fy, cy], 
                      [0, 0, 1]])
        CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height)
        
        # Clear any existing camera poses
        bpy.context.scene.frame_set(0)
        
        # Store projected keypoints for all frames and objects
        all_projected_keypoints = {}
        for obj_identifier in all_keypoints_3d.keys():
            all_projected_keypoints[obj_identifier] = {kp_name: [] for kp_name in all_keypoints_3d[obj_identifier].keys()}
        
        # Sample camera poses
        poses = 0
        tries = 0
        max_tries = 10000
        
        while tries < max_tries and poses < num_images:
            # Set random world lighting strength
            world = bpy.context.scene.world
            if world and world.node_tree and "Background" in world.node_tree.nodes:
                world.node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)
            
            # Set HDRI background if using HDRI
            if use_hdri and hdr_files:
                random_hdr_file = random.choice(hdr_files)
                bproc.world.set_world_background_hdr_img(
                random_hdr_file,
                strength=1.0,
                rotation_euler=[0.0, 0.0, random.uniform(0, 2 * np.pi)]
                )
                print(f"[HDRI] Using HDRI: {os.path.basename(random_hdr_file)}")
         
            # Sample camera location with limited elevation angles
            location = bproc.sampler.shell(
                center=scene_center,
                radius_min=3,  # Slightly further for multiple objects
                radius_max=12,
                elevation_min=-45,  # LIMITED: avoid extreme low angles
                elevation_max=75    # LIMITED: avoid extreme high angles
            )
            
            # Compute rotation with limited in-plane rotation
            lookat_point = scene_center + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                lookat_point - location, 
                inplane_rot=np.random.uniform(-0.3927, 0.3927)  # LIMITED: ±22.5 degrees
            )
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # Check if all objects are visible
            all_visible = all(obj_data['obj'] in bproc.camera.visible_objects(cam2world_matrix) 
                             for obj_data in loaded_objects)
            
            if all_visible:
                bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
                
                # Project keypoints for all objects
                for obj_identifier, keypoints_3d in all_keypoints_3d.items():
                    keypoint_coords_3d = np.array(list(keypoints_3d.values()))
                    projected_2d = project_points(keypoint_coords_3d, frame=poses)
                    
                    if projected_2d is not None and len(projected_2d) == len(keypoints_3d):
                        for i, kp_name in enumerate(keypoints_3d.keys()):
                            all_projected_keypoints[obj_identifier][kp_name].append(projected_2d[i])
                    else:
                        for kp_name in keypoints_3d.keys():
                            all_projected_keypoints[obj_identifier][kp_name].append(None)
                
                poses += 1
            
            tries += 1
        
        if poses < num_images:
            print(f"Warning: Only generated {poses}/{num_images} poses for multi-object scene")
        
        # Set rendering parameters
        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_output_format(enable_transparency=True)
        bproc.renderer.enable_segmentation_output(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": 0}
        )
        
        # Render
        print(f"Rendering {poses} images...")
        data = bproc.renderer.render()
        
        # Get keypoint colors
        keypoint_colors = KeypointDetector.get_keypoint_colors()
        
        # Process each rendered frame
        for i, image in enumerate(data["colors"]):
            # Generate base filename with all object identifiers
            obj_identifiers = [obj_data['identifier'] for obj_data in loaded_objects]
            combined_identifier = "_".join(obj_identifiers)
            image_filename = f"multi_{combined_identifier}_{i:06d}.png"
            viz_filename = f"vis_multi_{combined_identifier}_{i:06d}.png"
            
            # Collect all keypoints and bboxes for this frame
            frame_objects = []
            
            for obj_idx, obj_data in enumerate(loaded_objects):
                obj_identifier = obj_data['identifier']
                category = obj_data['category']
                original_name = obj_data['original_name']
                
                # Get keypoints for this object and frame - UPDATED TO USE VIEW-DEPENDENT LABELING
                frame_keypoints_raw = []
                for kp_name in all_keypoints_3d[obj_identifier].keys():
                    if i < len(all_projected_keypoints[obj_identifier][kp_name]):
                        pt = all_projected_keypoints[obj_identifier][kp_name][i]
                        frame_keypoints_raw.append(pt)
                    else:
                        frame_keypoints_raw.append(None)
                
                # Apply view-dependent left/right assignment
                frame_keypoints = KeypointDetector.assign_view_dependent_labels(
                    all_keypoints_3d[obj_identifier], frame_keypoints_raw, i
                )
                
                # Compute bbox from segmentation mask
                try:
                    instance_map = data["instance_segmaps"][i]
                    # For multi-object scenes, we need to identify which instance belongs to which object
                    # This is a simplified approach - you may need to refine based on your specific needs
                    unique_instances = np.unique(instance_map)
                    valid_instances = [inst for inst in unique_instances if inst > 0]  # Skip background
                    
                    if len(valid_instances) > obj_idx:
                        # Use the obj_idx-th valid instance for this object
                        instance_id = valid_instances[obj_idx]
                        obj_mask = (instance_map == instance_id).astype(np.uint8)
                        
                        if obj_mask.sum() > 0:
                            y_indices, x_indices = np.where(obj_mask)
                            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                        else:
                            bbox = [0.0, 0.0, 0.0, 0.0]
                    else:
                        bbox = [0.0, 0.0, 0.0, 0.0]
                        
                except Exception as e:
                    print(f"Warning: Could not compute bbox for {obj_identifier} frame {i}: {e}")
                    bbox = [0.0, 0.0, 0.0, 0.0]
                
                frame_objects.append({
                    "object_id": obj_identifier,
                    "category": category,
                    "original_name": original_name,
                    "keypoints": frame_keypoints,
                    "bbox": bbox
                })
            
            # Save base rendered image (always save this)
            if global_images_dir:
                image_path = os.path.join(global_images_dir, image_filename)
                plt.imsave(image_path, image)
                
                # Create multi-object visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(image)
                
                legend_patches = []
                
                # Plot keypoints and bboxes for all objects
                for obj_data in frame_objects:
                    obj_identifier = obj_data["object_id"]
                    keypoints_2d = obj_data["keypoints"]
                    bbox = obj_data["bbox"]
                    category = obj_data["category"]
                    
                    # Plot keypoints
                    for kp_name, kp_color in keypoint_colors.items():
                        if kp_name in keypoints_2d:
                            pt = keypoints_2d[kp_name]
                            if pt is not None:
                                x, y = pt
                                ax.plot(x, y, 'o', markersize=6, color=kp_color, 
                                       markeredgecolor='white', markeredgewidth=1)
                                # Only add to legend once per keypoint type
                                if not any(p.get_label() == kp_name for p in legend_patches):
                                    legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))
                    
                    # Plot bounding box
                    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        rect = mpatches.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1] - 5, f'{obj_identifier}', color='lime', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))
                
                # Add legend
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)
                
                ax.axis('off')
                title = f'Multi-Tool Scene - Frame {i} ({len(frame_objects)} objects)'
                ax.set_title(title, fontsize=10)
                plt.tight_layout()
                
                viz_path = os.path.join(global_viz_dir, viz_filename)
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to global annotations
                global_annotations["images"].append({
                    "file_name": image_filename,
                    "objects": frame_objects,
                    "scene_type": "multi_object",
                    "num_objects": len(frame_objects)
                })
            
            # Process background version if requested
            if use_background and backgrounds_dir and bg_images_dir:
                #bg_image = paste_background_on_image(image, backgrounds_dir)
                bg_image = enhance_background_integration(image, backgrounds_dir)
                # Save background image
                bg_image_path = os.path.join(bg_images_dir, image_filename)
                plt.imsave(bg_image_path, bg_image)
                
                # Create background visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(bg_image)
                
                legend_patches = []
                
                # Plot keypoints and bboxes for all objects
                for obj_data in frame_objects:
                    obj_identifier = obj_data["object_id"]
                    keypoints_2d = obj_data["keypoints"]
                    bbox = obj_data["bbox"]
                    
                    # Plot keypoints
                    for kp_name, kp_color in keypoint_colors.items():
                        if kp_name in keypoints_2d:
                            pt = keypoints_2d[kp_name]
                            if pt is not None:
                                x, y = pt
                                ax.plot(x, y, 'o', markersize=6, color=kp_color, 
                                       markeredgecolor='white', markeredgewidth=1)
                                if not any(p.get_label() == kp_name for p in legend_patches):
                                    legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))
                    
                    # Plot bounding box
                    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        rect = mpatches.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1] - 5, f'{obj_identifier}', color='lime', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))
                
                # Add legend
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)
                
                ax.axis('off')
                title = f'Multi-Tool Scene - Frame {i} (with Background, {len(frame_objects)} objects)'
                ax.set_title(title, fontsize=10)
                plt.tight_layout()
                
                bg_viz_path = os.path.join(bg_viz_dir, viz_filename)
                plt.savefig(bg_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to background annotations
                bg_annotations["images"].append({
                    "file_name": image_filename,
                    "objects": frame_objects,
                    "scene_type": "multi_object",
                    "num_objects": len(frame_objects)
                })
            
            # Process HDRI version if requested
            if use_hdri and hdri_images_dir and not (use_background and not use_hdri):
                # If we're only doing HDRI, the rendered image already has HDRI background
                hdri_image = image  # Use the already rendered HDRI image
                
                # Save HDRI image
                hdri_image_path = os.path.join(hdri_images_dir, image_filename)
                plt.imsave(hdri_image_path, hdri_image)
                
                # Create HDRI visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(hdri_image)
                
                legend_patches = []
                
                # Plot keypoints and bboxes for all objects
                for obj_data in frame_objects:
                    obj_identifier = obj_data["object_id"]
                    keypoints_2d = obj_data["keypoints"]
                    bbox = obj_data["bbox"]
                    
                    # Plot keypoints
                    for kp_name, kp_color in keypoint_colors.items():
                        if kp_name in keypoints_2d:
                            pt = keypoints_2d[kp_name]
                            if pt is not None:
                                x, y = pt
                                ax.plot(x, y, 'o', markersize=6, color=kp_color, 
                                       markeredgecolor='white', markeredgewidth=1)
                                if not any(p.get_label() == kp_name for p in legend_patches):
                                    legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))
                    
                    # Plot bounding box
                    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        rect = mpatches.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1] - 5, f'{obj_identifier}', color='lime', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))
                
                # Add legend
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)
                
                ax.axis('off')
                title = f'Multi-Tool Scene - Frame {i} (with HDRI, {len(frame_objects)} objects)'
                ax.set_title(title, fontsize=10)
                plt.tight_layout()
                
                hdri_viz_path = os.path.join(hdri_viz_dir, viz_filename)
                plt.savefig(hdri_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to HDRI annotations
                hdri_annotations["images"].append({
                    "file_name": image_filename,
                    "objects": frame_objects,
                    "scene_type": "multi_object",
                    "num_objects": len(frame_objects)
                })
        
        print(f"Successfully processed multi-object scene: {poses} images rendered")
        return True, combined_identifier, poses
        
    except Exception as e:
        print(f"Error processing multi-object scene: {e}")
        import traceback
        traceback.print_exc()
        return False, "multi_scene", 0

def process_single_object(obj_path, camera_params, base_output_dir, num_images=25, is_first_object=False, 
                         global_images_dir=None, global_viz_dir=None, global_annotations=None, object_counter=None,
                         use_background=False, use_hdri=False, backgrounds_dir=None, haven_path=None,
                         bg_images_dir=None, bg_viz_dir=None, hdri_images_dir=None, hdri_viz_dir=None,
                         bg_annotations=None, hdri_annotations=None):
    """Process a single object file and generate renders with category-specific keypoints."""
    
    # Extract object info from path
    obj_filename = os.path.basename(obj_path)
    obj_name = os.path.splitext(obj_filename)[0]
    
    # Determine category from parent folder
    parent_folder = os.path.basename(os.path.dirname(obj_path))
    if 'needle_holder' in parent_folder.lower():
        category = 'needle_holder'
        category_id = 1
        category_prefix = 'NH'
    elif 'tweezers' in parent_folder.lower():
        category = 'tweezers'
        category_id = 2
        category_prefix = 'TW'
    else:
        category = 'unknown'
        category_id = 0
        category_prefix = 'UK'
    
    # Generate object identifier
    obj_identifier = f"{obj_name}"
    object_counter['count'] += 1

    print(f"\nProcessing {category}/{obj_name} -> {obj_identifier} ({obj_filename})")
    
    # Load HDR files if needed
    hdr_files = []
    if use_hdri and haven_path:
        try:
            hdr_files = get_hdr_img_paths_from_haven(haven_path)
            if not hdr_files:
                print(f"Warning: No HDR files found in {haven_path}")
                use_hdri = False
        except Exception as e:
            print(f"Warning: Could not load HDR files: {e}")
            use_hdri = False
    
    try:
        # Clear the scene before loading new object (but not for the first object)
        if not is_first_object:
            bproc.clean_up(clean_up_camera=False)
        
        # Ensure world exists and has proper node setup
        world = bpy.context.scene.world
        if world is None or world.node_tree is None:
            if "World" in bpy.data.worlds:
                world = bpy.data.worlds["World"]
            else:
                world = bpy.data.worlds.new("World")
            
            bpy.context.scene.world = world
            world.use_nodes = True
            
            if not world.node_tree.nodes:
                background_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
                output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
                world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        
        # Load the object
        obj = bproc.loader.load_obj(obj_path)[0]
        obj.set_cp("category_id", category_id)
        
        # FIXED: Handle materials with proper opacity settings
        materials = obj.get_materials()
        if len(materials) > 0:
            mat = materials[0]
            
            # Basic material properties
            safe_set_principled_shader_value(mat, "IOR", random.uniform(1.0, 2.5))
            safe_set_principled_shader_value(mat, "Roughness", random.uniform(0.1, 0.8))
            safe_set_principled_shader_value(mat, "Metallic", 1)
            
            # CRITICAL: Ensure full opacity
            safe_set_principled_shader_value(mat, "Alpha", 1.0)
            safe_set_principled_shader_value(mat, "Transmission", 0.0)
            
            # Set blend mode to opaque to prevent transparency issues
            try:
                if hasattr(mat.blender_obj, 'blend_method'):
                    mat.blender_obj.blend_method = 'OPAQUE'
                elif hasattr(mat.blender_obj, 'use_transparency'):
                    mat.blender_obj.use_transparency = False
            except Exception as e:
                print(f"Warning: Could not set blend mode: {e}")
            
            # Handle second material if present
            if len(materials) > 1:
                mat2 = materials[1]
                random_gold_hsv_color = np.random.uniform([0.03, 0.95, 0.8], [0.25, 1.0, 1.0])
                random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0]
                
                safe_set_principled_shader_value(mat2, "Base Color", random_gold_color)
                safe_set_principled_shader_value(mat2, "IOR", random.uniform(1.0, 2.5))
                safe_set_principled_shader_value(mat2, "Roughness", random.uniform(0.1, 0.8))
                safe_set_principled_shader_value(mat2, "Metallic", 1)
                
                # CRITICAL: Ensure full opacity
                safe_set_principled_shader_value(mat2, "Alpha", 1.0)
                safe_set_principled_shader_value(mat2, "Transmission", 0.0)
                
                # Set blend mode to opaque
                try:
                    if hasattr(mat2.blender_obj, 'blend_method'):
                        mat2.blender_obj.blend_method = 'OPAQUE'
                    elif hasattr(mat2.blender_obj, 'use_transparency'):
                        mat2.blender_obj.use_transparency = False
                except Exception as e:
                    print(f"Warning: Could not set blend mode for material 2: {e}")
        
        # Create lighting
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(bproc.sampler.shell(
            center=obj.get_location(),
            radius_min=1,
            radius_max=5,
            elevation_min=1,
            elevation_max=89
        ))
        light.set_energy(random.uniform(100, 1000))
        
        # Set camera intrinsics
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
        im_width = camera_params["width"]
        im_height = camera_params["height"]
        K = np.array([[fx, 0, cx], 
                      [0, fy, cy], 
                      [0, 0, 1]])
        CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height)
        
        # Clear any existing camera poses
        bpy.context.scene.frame_set(0)
        
        # Get mesh vertices in world coordinates
        mesh = obj.get_mesh()
        verts = mesh.vertices
        obj2world = obj.get_local2world_mat()
        verts_world = [obj2world @ np.append(v.co, 1.0) for v in verts]
        verts_world = [v[:3] for v in verts_world]
        
        # Detect keypoints based on category
       

        # Detect keypoints based on category
        if category == 'needle_holder':
            keypoints_3d = KeypointDetector.detect_needle_holder_keypoints(verts_world)
            # REMOVED: No longer delete joint_center keypoint - it's now intentionally included
        elif category == 'tweezers':
            keypoints_3d = KeypointDetector.detect_tweezers_keypoints(verts_world)
        else:
            # Fallback to standard corners only
            verts_array = np.array(verts_world)
            keypoints_3d = KeypointDetector._get_standard_corners_actual_verts(
                verts_array, verts_array[:, 0], verts_array[:, 1], verts_array[:, 2]
            )
        
        print(f"Detected keypoints for {category}:")
        for kp_name, kp_coord in keypoints_3d.items():
            print(f"  {kp_name}: {kp_coord}")
        
        # Store projected keypoints for all frames
        all_projected_keypoints = {kp_name: [] for kp_name in keypoints_3d.keys()}
        
        # Sample camera poses
        poses = 0
        tries = 0
        max_tries = 10000
        
        while tries < max_tries and poses < num_images:
            # Set random world lighting strength
            world = bpy.context.scene.world
            if world and world.node_tree and "Background" in world.node_tree.nodes:
                world.node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)
            
            # Set HDRI background if using HDRI
            if use_hdri and hdr_files:
                random_hdr_file = random.choice(hdr_files)
                bproc.world.set_world_background_hdr_img(
                random_hdr_file,
                strength=1.0,
                rotation_euler=[0.0, 0.0, random.uniform(0, 2 * np.pi)]
                )
                print(f"[HDRI] Using HDRI: {os.path.basename(random_hdr_file)}")
         
            # Sample camera location
            location = bproc.sampler.shell(
                center=obj.get_location(),
                radius_min=2,
                radius_max=10,
                elevation_min=-45,
                elevation_max=75
            )
            
            # Compute rotation
            lookat_point = obj.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                lookat_point - location, 
                inplane_rot=np.random.uniform(-0.3927, 0.3927)
            )
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # Check if object is visible
            if obj in bproc.camera.visible_objects(cam2world_matrix):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
                
                # Project all keypoints to 2D
                keypoint_coords_3d = np.array(list(keypoints_3d.values()))
                projected_2d = project_points(keypoint_coords_3d, frame=poses)
                
                # Store projected keypoints
                if projected_2d is not None and len(projected_2d) == len(keypoints_3d):
                    for i, kp_name in enumerate(keypoints_3d.keys()):
                        all_projected_keypoints[kp_name].append(projected_2d[i])
                else:
                    # Store None for missing projections
                    for kp_name in keypoints_3d.keys():
                        all_projected_keypoints[kp_name].append(None)
                
                poses += 1
            
            tries += 1
        
        if poses < num_images:
            print(f"Warning: Only generated {poses}/{num_images} poses for {obj_name}")
        
        # Set rendering parameters
        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_output_format(enable_transparency=True)
        bproc.renderer.enable_segmentation_output(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": 0}
        )
        
        # Render
        print(f"Rendering {poses} images...")
        data = bproc.renderer.render()
        
        # Get keypoint colors
        keypoint_colors = KeypointDetector.get_keypoint_colors()
        
        # Process each rendered frame
        for i, image in enumerate(data["colors"]):
            # Generate base filenames
            image_filename = f"{obj_identifier}_{i:06d}.png"
            viz_filename = f"vis_{obj_identifier}_{i:06d}.png"
            
            # Get keypoints for this frame and apply view-dependent labeling
            frame_keypoints_raw = []
            for kp_name in keypoints_3d.keys():
                if i < len(all_projected_keypoints[kp_name]):
                    pt = all_projected_keypoints[kp_name][i]
                    frame_keypoints_raw.append(pt)
                else:
                    frame_keypoints_raw.append(None)
            
            # Apply view-dependent left/right assignment
            frame_keypoints = KeypointDetector.assign_view_dependent_labels(
                keypoints_3d, frame_keypoints_raw, i
            )
            
            # Compute bbox from segmentation mask
            try:
                instance_map = data["instance_segmaps"][i]
                obj_mask = (instance_map == 1).astype(np.uint8)
                if obj_mask.sum() > 0:
                    y_indices, x_indices = np.where(obj_mask)
                    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
            except Exception as e:
                print(f"Warning: Could not compute bbox for frame {i}: {e}")
                bbox = [0.0, 0.0, 0.0, 0.0]
            
            # Save base rendered image (always save this)
            if global_images_dir:
                image_path = os.path.join(global_images_dir, image_filename)
                plt.imsave(image_path, image)
                
                # Create base visualization
                fig = create_visualization(image, frame_keypoints, keypoint_colors, obj_identifier, category, i, bbox=bbox)
                viz_path = os.path.join(global_viz_dir, viz_filename)
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to global annotations
                global_annotations["images"].append({
                    "file_name": image_filename,
                    "keypoints": frame_keypoints,
                    "bbox": bbox,
                    "object_id": obj_identifier,
                    "category": category,
                    "original_name": obj_name
                })
            
            # Process background version if requested
            if use_background and backgrounds_dir and bg_images_dir:
                #bg_image = paste_background_on_image(image, backgrounds_dir)
                bg_image = enhance_background_integration(image, backgrounds_dir)
                # Save background image
                bg_image_path = os.path.join(bg_images_dir, image_filename)
                plt.imsave(bg_image_path, bg_image)
                
                # Create background visualization
                fig = create_visualization(bg_image, frame_keypoints, keypoint_colors, obj_identifier, category, i, title_suffix=" (with Background)", bbox=bbox)
                bg_viz_path = os.path.join(bg_viz_dir, viz_filename)
                plt.savefig(bg_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to background annotations
                bg_annotations["images"].append({
                    "file_name": image_filename,
                    "keypoints": frame_keypoints,
                    "bbox": bbox,
                    "object_id": obj_identifier,
                    "category": category,
                    "original_name": obj_name
                })
            
            # Process HDRI version if requested (and not already done)
            if use_hdri and hdri_images_dir and not (use_background and not use_hdri):
                # If we're only doing HDRI, the rendered image already has HDRI background
                hdri_image = image  # Use the already rendered HDRI image
                
                # Save HDRI image
                hdri_image_path = os.path.join(hdri_images_dir, image_filename)
                plt.imsave(hdri_image_path, hdri_image)
                
                # Create HDRI visualization
                fig = create_visualization(image, frame_keypoints, keypoint_colors, obj_identifier, category, i, bbox=bbox)
                hdri_viz_path = os.path.join(hdri_viz_dir, viz_filename)
                plt.savefig(hdri_viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to HDRI annotations
                hdri_annotations["images"].append({
                    "file_name": image_filename,
                    "keypoints": frame_keypoints,
                    "bbox": bbox,
                    "object_id": obj_identifier,
                    "category": category,
                    "original_name": obj_name
                })
        
        print(f"Successfully processed {obj_identifier}: {poses} images rendered")
        return True, obj_identifier, poses
        
    except Exception as e:
        print(f"Error processing {obj_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, obj_identifier, 0

def main():
    parser = argparse.ArgumentParser(description='Batch render all surgical tools with category-specific keypoints')
    parser.add_argument('--tools_dir', 
                        default="/datashare/project/surgical_tools_models", 
                        help="Path to surgical tools models directory")
    parser.add_argument('--camera_params', 
                        default="/datashare/project/camera.json", 
                        help="Camera intrinsics in json format")
    parser.add_argument('--output_dir', 
                        default="output_all_tools_multi_tool_hdri_test", 
                        help="Base output directory")
    parser.add_argument('--num_images', 
                        type=int, default=1, 
                        help="Number of images per object")
    parser.add_argument('--categories',
                        nargs='+',
                        default=['needle_holder', 'tweezers'],
                        help="Categories to process")
    
    # Background and HDRI options
    parser.add_argument('--use_background', action='store_true',
                        help="Enable background pasting")
    parser.add_argument('--use_hdri', action='store_true',
                        help="Enable HDRI backgrounds")
    parser.add_argument('--backgrounds_dir',
                        default="/datashare/project/train2017",
                        help="Directory containing background images")
    parser.add_argument('--haven_path',
                        default="/datashare/project/haven/",
                        help="Path to the haven HDRI images")
    
    # NEW: Multi-tool option
    parser.add_argument('--use_multi_tools', action='store_true',
                        help="Enable rendering multiple tools in the same scene")
    parser.add_argument('--multi_tool_ratio', type=float, default=0.7,
                        help="Ratio of multi-tool scenes (0.0-1.0, default: 0.3)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_background and not os.path.exists(args.backgrounds_dir):
        print(f"Error: Background directory does not exist: {args.backgrounds_dir}")
        return
    
    if args.use_hdri and not os.path.exists(args.haven_path):
        print(f"Error: Haven path does not exist: {args.haven_path}")
        return
    
    if not args.use_background and not args.use_hdri:
        print("Note: Neither --use_background nor --use_hdri specified. Will render base images only.")
    
    # Validate multi-tool ratio
    if args.multi_tool_ratio < 0.0 or args.multi_tool_ratio > 1.0:
        print("Error: --multi_tool_ratio must be between 0.0 and 1.0")
        return
    
    # Create base output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create base subdirectories (always created)
    global_images_dir = os.path.join(args.output_dir, 'images')
    global_viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(global_images_dir, exist_ok=True)
    os.makedirs(global_viz_dir, exist_ok=True)
    
    # Create background subdirectories if needed
    bg_images_dir = None
    bg_viz_dir = None
    if args.use_background:
        bg_images_dir = os.path.join(args.output_dir, 'images_with_background')
        bg_viz_dir = os.path.join(args.output_dir, 'visualizations_with_background')
        os.makedirs(bg_images_dir, exist_ok=True)
        os.makedirs(bg_viz_dir, exist_ok=True)
    
    # Create HDRI subdirectories if needed
    hdri_images_dir = None
    hdri_viz_dir = None
    if args.use_hdri:
        hdri_images_dir = os.path.join(args.output_dir, 'images_with_hdri')
        hdri_viz_dir = os.path.join(args.output_dir, 'visualizations_with_hdri')
        os.makedirs(hdri_images_dir, exist_ok=True)
        os.makedirs(hdri_viz_dir, exist_ok=True)
    
    # Initialize global annotations
    global_annotations = {"images": []}
    bg_annotations = {"images": []} if args.use_background else None
    hdri_annotations = {"images": []} if args.use_hdri else None
    
    # Object counter for generating identifiers
    object_counter = {"count": 1}
    
    # Load camera parameters once
    with open(args.camera_params, "r") as file:
        camera_params = json.load(file)
    
    # Find all object files
    all_objects = []
    
    for category in args.categories:
        category_path = os.path.join(args.tools_dir, category)
        if os.path.exists(category_path):
            obj_files = glob.glob(os.path.join(category_path, "*.obj"))
            all_objects.extend(obj_files)
            print(f"Found {len(obj_files)} objects in {category}")
        else:
            print(f"Warning: Category directory not found: {category_path}")
    
    print(f"\nTotal objects to process: {len(all_objects)}")
    
    # Print rendering configuration
    print(f"\nRendering Configuration:")
    print(f"  Base images: Always enabled")
    print(f"  Background pasting: {'Enabled' if args.use_background else 'Disabled'}")
    print(f"  HDRI backgrounds: {'Enabled' if args.use_hdri else 'Disabled'}")
    print(f"  Multi-tool scenes: {'Enabled' if args.use_multi_tools else 'Disabled'}")
    if args.use_multi_tools:
        print(f"  Multi-tool ratio: {args.multi_tool_ratio:.1%}")
    
    # Initialize BlenderProc once at the beginning
    print("Initializing BlenderProc...")
    bproc.init()
    
    try:
        # Process objects
        successful_objects = []
        failed_objects = []
        processed_objects = set()  # Keep track of processed objects to avoid duplicates
        
        # Calculate how many multi-tool scenes to generate
        if args.use_multi_tools and len(all_objects) >= 2:
            total_scenes = len(all_objects)
            num_multi_scenes = int(total_scenes * args.multi_tool_ratio)
            num_single_scenes = total_scenes - num_multi_scenes
            
            print(f"\nScene distribution:")
            print(f"  Single-tool scenes: {num_single_scenes}")
            print(f"  Multi-tool scenes: {num_multi_scenes}")
            
            scene_count = 0
            
            # Generate multi-tool scenes first
            for multi_scene_idx in range(num_multi_scenes):
                # Select available objects for multi-tool scene
                available_objects = [obj for obj in all_objects if obj not in processed_objects]
                if len(available_objects) < 2:
                    print(f"Warning: Not enough objects left for multi-tool scene {multi_scene_idx + 1}")
                    break
                
                scene_count += 1
                print(f"\n{'='*60}")
                print(f"Processing multi-tool scene {scene_count}/{num_multi_scenes + num_single_scenes}")
                print(f"{'='*60}")
                
                success, scene_identifier, num_rendered = process_multi_objects(
                    available_objects,
                    camera_params,
                    args.output_dir,
                    args.num_images,
                    global_images_dir=global_images_dir,
                    global_viz_dir=global_viz_dir,
                    global_annotations=global_annotations,
                    object_counter=object_counter,
                    use_background=args.use_background,
                    use_hdri=args.use_hdri,
                    backgrounds_dir=args.backgrounds_dir if args.use_background else None,
                    haven_path=args.haven_path if args.use_hdri else None,
                    bg_images_dir=bg_images_dir,
                    bg_viz_dir=bg_viz_dir,
                    hdri_images_dir=hdri_images_dir,
                    hdri_viz_dir=hdri_viz_dir,
                    bg_annotations=bg_annotations,
                    hdri_annotations=hdri_annotations
                )
                
                if success:
                    successful_objects.append((scene_identifier, num_rendered))
                    # Mark some objects as processed (estimate 1-2 objects per multi-scene)
                    num_to_mark = min(2, len(available_objects))
                    for obj in available_objects[:num_to_mark]:
                        processed_objects.add(obj)
                else:
                    failed_objects.append(f"multi_scene_{multi_scene_idx + 1}")
            
            # Generate remaining single-tool scenes
            available_objects = [obj for obj in all_objects if obj not in processed_objects]
            remaining_to_process = min(num_single_scenes, len(available_objects))
            
            for i, obj_path in enumerate(available_objects[:remaining_to_process]):
                obj_name = os.path.splitext(os.path.basename(obj_path))[0]
                scene_count += 1
                print(f"\n{'='*60}")
                print(f"Processing single-tool scene {scene_count}/{num_multi_scenes + num_single_scenes}: {obj_name}")
                print(f"{'='*60}")
                
                success, obj_identifier, num_rendered = process_single_object(
                    obj_path,
                    camera_params,
                    args.output_dir,
                    args.num_images,
                    is_first_object=False,  # BlenderProc already initialized
                    global_images_dir=global_images_dir,
                    global_viz_dir=global_viz_dir,
                    global_annotations=global_annotations,
                    object_counter=object_counter,
                    use_background=args.use_background,
                    use_hdri=args.use_hdri,
                    backgrounds_dir=args.backgrounds_dir if args.use_background else None,
                    haven_path=args.haven_path if args.use_hdri else None,
                    bg_images_dir=bg_images_dir,
                    bg_viz_dir=bg_viz_dir,
                    hdri_images_dir=hdri_images_dir,
                    hdri_viz_dir=hdri_viz_dir,
                    bg_annotations=bg_annotations,
                    hdri_annotations=hdri_annotations
                )
                
                if success:
                    successful_objects.append((obj_identifier, num_rendered))
                else:
                    failed_objects.append(obj_name)
        
        else:
            # Original single-object processing
            for i, obj_path in enumerate(all_objects, 1):
                obj_name = os.path.splitext(os.path.basename(obj_path))[0]
                print(f"\n{'='*60}")
                print(f"Processing object {i}/{len(all_objects)}: {obj_name}")
                print(f"{'='*60}")
                
                success, obj_identifier, num_rendered = process_single_object(
                    obj_path,
                    camera_params,
                    args.output_dir,
                    args.num_images,
                    is_first_object=(i == 1),
                    global_images_dir=global_images_dir,
                    global_viz_dir=global_viz_dir,
                    global_annotations=global_annotations,
                    object_counter=object_counter,
                    use_background=args.use_background,
                    use_hdri=args.use_hdri,
                    backgrounds_dir=args.backgrounds_dir if args.use_background else None,
                    haven_path=args.haven_path if args.use_hdri else None,
                    bg_images_dir=bg_images_dir,
                    bg_viz_dir=bg_viz_dir,
                    hdri_images_dir=hdri_images_dir,
                    hdri_viz_dir=hdri_viz_dir,
                    bg_annotations=bg_annotations,
                    hdri_annotations=hdri_annotations
                )
                
                if success:
                    successful_objects.append((obj_identifier, num_rendered))
                else:
                    failed_objects.append(obj_name)
        
        # Save all annotation files
        annotations_file = os.path.join(args.output_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(global_annotations, f, indent=2)
        print(f"\nBase annotations saved to: {annotations_file}")
        
        if args.use_background and bg_annotations:
            bg_annotations_file = os.path.join(args.output_dir, "annotations_with_background.json")
            with open(bg_annotations_file, 'w') as f:
                json.dump(bg_annotations, f, indent=2)
            print(f"Background annotations saved to: {bg_annotations_file}")
        
        if args.use_hdri and hdri_annotations:
            hdri_annotations_file = os.path.join(args.output_dir, "annotations_with_hdri.json")
            with open(hdri_annotations_file, 'w') as f:
                json.dump(hdri_annotations, f, indent=2)
            print(f"HDRI annotations saved to: {hdri_annotations_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {len(successful_objects)}")
        print(f"Failed: {len(failed_objects)}")
        
        # Count total images generated
        total_base_images = len(global_annotations['images'])
        total_bg_images = len(bg_annotations['images']) if bg_annotations else 0
        total_hdri_images = len(hdri_annotations['images']) if hdri_annotations else 0
        
        print(f"Base images generated: {total_base_images}")
        if args.use_background:
            print(f"Background images generated: {total_bg_images}")
        if args.use_hdri:
            print(f"HDRI images generated: {total_hdri_images}")
        
        # Count scene types if multi-tools was used
        if args.use_multi_tools:
            single_scenes = sum(1 for img in global_annotations['images'] if 'scene_type' not in img or img.get('scene_type') != 'multi_object')
            multi_scenes = sum(1 for img in global_annotations['images'] if img.get('scene_type') == 'multi_object')
            print(f"Single-tool scenes: {single_scenes}")
            print(f"Multi-tool scenes: {multi_scenes}")
        
        print(f"\nOutput directories:")
        print(f"  Base images: {global_images_dir}")
        print(f"  Base visualizations: {global_viz_dir}")
        if args.use_background:
            print(f"  Background images: {bg_images_dir}")
            print(f"  Background visualizations: {bg_viz_dir}")
        if args.use_hdri:
            print(f"  HDRI images: {hdri_images_dir}")
            print(f"  HDRI visualizations: {hdri_viz_dir}")
        
        if successful_objects:
            print("\nSuccessful objects:")
            for obj_identifier, num_rendered in successful_objects:
                print(f"  ✓ {obj_identifier}: {num_rendered} images")
        
        if failed_objects:
            print("\nFailed objects:")
            for obj_name in failed_objects:
                print(f"  ✗ {obj_name}")
        
        # Create a comprehensive summary file
        summary_file = os.path.join(args.output_dir, "processing_summary.json")
        summary_data = {
            "configuration": {
                "use_background": args.use_background,
                "use_hdri": args.use_hdri,
                "use_multi_tools": args.use_multi_tools,
                "multi_tool_ratio": args.multi_tool_ratio if args.use_multi_tools else 0.0,
                "backgrounds_dir": args.backgrounds_dir if args.use_background else None,
                "haven_path": args.haven_path if args.use_hdri else None,
                "num_images_per_object": args.num_images,
                "categories": args.categories
            },
            "results": {
                "total_objects": len(all_objects),
                "successful": len(successful_objects),
                "failed": len(failed_objects),
                "base_images": total_base_images,
                "background_images": total_bg_images,
                "hdri_images": total_hdri_images
            },
            "output_structure": {
                "base": {
                    "images": global_images_dir,
                    "visualizations": global_viz_dir,
                    "annotations": annotations_file
                }
            },
            "successful_objects": [
                {"identifier": identifier, "images_rendered": num_rendered}
                for identifier, num_rendered in successful_objects
            ],
            "failed_objects": failed_objects
        }
        
        # Add scene type breakdown if multi-tools was used
        if args.use_multi_tools:
            single_scenes = sum(1 for img in global_annotations['images'] if 'scene_type' not in img or img.get('scene_type') != 'multi_object')
            multi_scenes = sum(1 for img in global_annotations['images'] if img.get('scene_type') == 'multi_object')
            summary_data["results"]["single_tool_scenes"] = single_scenes
            summary_data["results"]["multi_tool_scenes"] = multi_scenes
        
        # Add background and HDRI output info if used
        if args.use_background:
            summary_data["output_structure"]["background"] = {
                "images": bg_images_dir,
                "visualizations": bg_viz_dir,
                "annotations": os.path.join(args.output_dir, "annotations_with_background.json")
            }
        
        if args.use_hdri:
            summary_data["output_structure"]["hdri"] = {
                "images": hdri_images_dir,
                "visualizations": hdri_viz_dir,
                "annotations": os.path.join(args.output_dir, "annotations_with_hdri.json")
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nComprehensive summary saved to: {summary_file}")
    
    finally:
        # Clean up BlenderProc at the very end
        bproc.clean_up()

if __name__ == "__main__":
    main()