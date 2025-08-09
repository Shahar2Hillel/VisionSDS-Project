# Fixed BlenderProc script with unified processing and background-only output
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
        Detect keypoints specific to needle holders with consistent anatomical labeling.
        Uses PCA to establish consistent anatomical coordinate system.
        """
        verts_array = np.array(verts_world)
        
        keypoints = {}
        
        # Step 1: Find the main axis using PCA
        centered_verts = verts_array - np.mean(verts_array, axis=0)
        cov_matrix = np.cov(centered_verts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue magnitude to get principal axes
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_axis = eigenvectors[:, sorted_indices[0]]  # Main elongation direction
        secondary_axis = eigenvectors[:, sorted_indices[1]]  # Width direction
        
        # Step 2: Project vertices onto principal axes
        projections_main = np.dot(centered_verts, principal_axis)
        projections_secondary = np.dot(centered_verts, secondary_axis)
        
        # Step 3: Define anatomical regions consistently
        # Find tip region (top 15% along main axis)
        tip_threshold = np.percentile(projections_main, 85)
        tip_indices = np.where(projections_main >= tip_threshold)[0]
        
        # Find handle region (bottom 15% along main axis)
        handle_threshold = np.percentile(projections_main, 15)
        handle_indices = np.where(projections_main <= handle_threshold)[0]
        
        # Step 4: Detect tips with consistent left/right orientation
        if len(tip_indices) > 0:
            tip_secondary_proj = projections_secondary[tip_indices]
            
            # Anatomical left tip = most negative secondary axis projection
            anatomical_left_tip_idx = tip_indices[np.argmin(tip_secondary_proj)]
            keypoints['top_left'] = verts_array[anatomical_left_tip_idx]
            
            # Anatomical right tip = most positive secondary axis projection  
            anatomical_right_tip_idx = tip_indices[np.argmax(tip_secondary_proj)]
            keypoints['top_right'] = verts_array[anatomical_right_tip_idx]
        else:
            # Fallback: use overall extremes
            main_max_idx = np.argmax(projections_main)
            keypoints['top_left'] = verts_array[main_max_idx]
            keypoints['top_right'] = verts_array[main_max_idx]
        
        # Step 5: Detect handle corners
        if len(handle_indices) > 0:
            handle_secondary_proj = projections_secondary[handle_indices]
            
            # Anatomical left handle = most negative secondary axis projection
            anatomical_left_handle_idx = handle_indices[np.argmin(handle_secondary_proj)]
            keypoints['bottom_left'] = verts_array[anatomical_left_handle_idx]
            
            # Anatomical right handle = most positive secondary axis projection
            anatomical_right_handle_idx = handle_indices[np.argmax(handle_secondary_proj)]
            keypoints['bottom_right'] = verts_array[anatomical_right_handle_idx]
        else:
            # Fallback
            main_min_idx = np.argmin(projections_main)
            keypoints['bottom_left'] = verts_array[main_min_idx]
            keypoints['bottom_right'] = verts_array[main_min_idx]
        
        # Step 6: Detect middle jaw points
        jaw_min = np.percentile(projections_main, 30)
        jaw_max = np.percentile(projections_main, 70)
        jaw_indices = np.where((projections_main >= jaw_min) & (projections_main <= jaw_max))[0]
        
        if len(jaw_indices) > 0:
            jaw_secondary_proj = projections_secondary[jaw_indices]
            secondary_median = np.median(jaw_secondary_proj)
            
            # Left jaw: points with secondary projection < median
            left_jaw_mask = jaw_secondary_proj <= secondary_median
            left_jaw_indices = jaw_indices[left_jaw_mask]
            
            if len(left_jaw_indices) > 0:
                left_jaw_points = verts_array[left_jaw_indices]
                left_centroid = np.mean(left_jaw_points, axis=0)
                distances = np.linalg.norm(left_jaw_points - left_centroid, axis=1)
                best_left_idx = left_jaw_indices[np.argmin(distances)]
                keypoints['middle_left'] = verts_array[best_left_idx]
            
            # Right jaw: points with secondary projection > median
            right_jaw_mask = jaw_secondary_proj >= secondary_median
            right_jaw_indices = jaw_indices[right_jaw_mask]
            
            if len(right_jaw_indices) > 0:
                right_jaw_points = verts_array[right_jaw_indices]
                right_centroid = np.mean(right_jaw_points, axis=0)
                distances = np.linalg.norm(right_jaw_points - right_centroid, axis=1)
                best_right_idx = right_jaw_indices[np.argmin(distances)]
                keypoints['middle_right'] = verts_array[best_right_idx]
        
        # Fallback for middle points if needed
        if 'middle_left' not in keypoints or 'middle_right' not in keypoints:
            secondary_median = np.median(projections_secondary)
            left_mask = projections_secondary <= secondary_median
            right_mask = projections_secondary >= secondary_median
            
            if 'middle_left' not in keypoints and np.any(left_mask):
                left_indices = np.where(left_mask)[0]
                left_points = verts_array[left_indices]
                left_centroid = np.mean(left_points, axis=0)
                distances = np.linalg.norm(left_points - left_centroid, axis=1)
                keypoints['middle_left'] = left_points[np.argmin(distances)]
            
            if 'middle_right' not in keypoints and np.any(right_mask):
                right_indices = np.where(right_mask)[0]
                right_points = verts_array[right_indices]
                right_centroid = np.mean(right_points, axis=0)
                distances = np.linalg.norm(right_points - right_centroid, axis=1)
                keypoints['middle_right'] = right_points[np.argmin(distances)]
        
        # Step 7: Joint center calculation (unchanged)
        if all(kp in keypoints for kp in ['top_left', 'top_right', 'middle_left', 'middle_right']):
            left_tip = keypoints['top_left']
            right_tip = keypoints['top_right']
            left_middle = keypoints['middle_left']
            right_middle = keypoints['middle_right']
            
            tips_center = (left_tip + right_tip) / 2.0
            shaft_center = (left_middle + right_middle) / 2.0
            
            joint_offset_ratio = 0.3
            joint_position = tips_center + joint_offset_ratio * (shaft_center - tips_center)
            
            distances_to_joint = np.linalg.norm(verts_array - joint_position, axis=1)
            closest_joint_idx = np.argmin(distances_to_joint)
            keypoints['joint_center'] = verts_array[closest_joint_idx]
        else:
            # Fallback joint calculation
            joint_z_max = np.percentile(projections_main, 40)
            joint_candidates_indices = np.where(projections_main <= joint_z_max)[0]
            
            if len(joint_candidates_indices) > 0:
                joint_candidates = verts_array[joint_candidates_indices]
                joint_secondary_proj = projections_secondary[joint_candidates_indices]
                
                # Find point closest to secondary axis center
                secondary_center = np.median(projections_secondary)
                secondary_distances = np.abs(joint_secondary_proj - secondary_center)
                
                center_threshold = np.percentile(secondary_distances, 30)
                centered_mask = secondary_distances <= center_threshold
                
                if np.any(centered_mask):
                    centered_candidates = joint_candidates[centered_mask]
                    centered_indices = joint_candidates_indices[centered_mask]
                    
                    # Among centered points, find closest to main axis median
                    main_center = np.median(projections_main[centered_indices])
                    main_distances = np.abs(projections_main[centered_indices] - main_center)
                    best_joint_local_idx = np.argmin(main_distances)
                    best_joint_idx = centered_indices[best_joint_local_idx]
                    
                    keypoints['joint_center'] = verts_array[best_joint_idx]
                else:
                    best_joint_local_idx = np.argmin(secondary_distances)
                    best_joint_idx = joint_candidates_indices[best_joint_local_idx]
                    keypoints['joint_center'] = verts_array[best_joint_idx]
            else:
                keypoints['joint_center'] = verts_array[np.argmin(projections_main)]
        
        return keypoints
        
    @staticmethod
    def detect_tweezers_keypoints(verts_world):
        """
        Detect keypoints specific to tweezers with consistent anatomical labeling.
        Uses PCA to establish consistent anatomical coordinate system.
        """
        verts_array = np.array(verts_world)
        
        # Use PCA to find principal axes
        centered_verts = verts_array - np.mean(verts_array, axis=0)
        cov_matrix = np.cov(centered_verts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_axis = eigenvectors[:, sorted_indices[0]]  # Main elongation direction
        secondary_axis = eigenvectors[:, sorted_indices[1]]  # Width direction
        
        projections_main = np.dot(centered_verts, principal_axis)
        projections_secondary = np.dot(centered_verts, secondary_axis)
        
        keypoints = {}
        
        # Bottom tip: absolute minimum along main axis (pinching end)
        min_main_idx = np.argmin(projections_main)
        keypoints['bottom_tip'] = verts_array[min_main_idx]
        
        # Find handle region (top 20% along main axis)
        handle_threshold = np.percentile(projections_main, 80)
        handle_indices = np.where(projections_main >= handle_threshold)[0]
        
        if len(handle_indices) > 0:
            handle_secondary_proj = projections_secondary[handle_indices]
            
            # Anatomical left handle = most negative secondary projection
            left_handle_idx = handle_indices[np.argmin(handle_secondary_proj)]
            keypoints['top_left'] = verts_array[left_handle_idx]
            
            # Anatomical right handle = most positive secondary projection
            right_handle_idx = handle_indices[np.argmax(handle_secondary_proj)]
            keypoints['top_right'] = verts_array[right_handle_idx]
        else:
            # Fallback: use overall extremes
            max_main_idx = np.argmax(projections_main)
            keypoints['top_left'] = verts_array[max_main_idx]
            keypoints['top_right'] = verts_array[max_main_idx]
        
        # Middle points: median region along main axis
        median_main = np.median(projections_main)
        middle_tolerance = np.std(projections_main) * 0.5
        middle_indices = np.where(np.abs(projections_main - median_main) <= middle_tolerance)[0]
        
        if len(middle_indices) > 0:
            middle_secondary_proj = projections_secondary[middle_indices]
            
            # Anatomical left middle = most negative secondary projection  
            left_mid_idx = middle_indices[np.argmin(middle_secondary_proj)]
            keypoints['mid_left'] = verts_array[left_mid_idx]
            
            # Anatomical right middle = most positive secondary projection
            right_mid_idx = middle_indices[np.argmax(middle_secondary_proj)]
            keypoints['mid_right'] = verts_array[right_mid_idx]
        else:
            # Fallback: use secondary axis extremes from all vertices
            left_overall_idx = np.argmin(projections_secondary)
            right_overall_idx = np.argmax(projections_secondary)
            keypoints['mid_left'] = verts_array[left_overall_idx]
            keypoints['mid_right'] = verts_array[right_overall_idx]
        
        return keypoints
    
    @staticmethod
    def assign_view_dependent_labels(keypoints_3d, keypoints_2d_projected, frame_idx):
        """
        FIXED: No more swapping based on view! 
        The 3D detection already ensures consistent anatomical labeling.
        Just convert to proper format without any left/right swapping.
        """
        if not keypoints_2d_projected or len(keypoints_2d_projected) == 0:
            return keypoints_2d_projected
        
        # Create mapping from 3D keypoint names to 2D projections
        kp_2d_dict = {}
        kp_names = list(keypoints_3d.keys())
        
        for i, kp_name in enumerate(kp_names):
            if i < len(keypoints_2d_projected) and keypoints_2d_projected[i] is not None:
                pt = keypoints_2d_projected[i]
                kp_2d_dict[kp_name] = [float(pt[0]), float(pt[1])]
        
        # Return the mapping without any swapping
        # The anatomical consistency is maintained by the 3D detection using PCA
        return kp_2d_dict
    
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

def process_objects(obj_paths, camera_params, output_dir, num_images=25, 
                   object_counter=None, backgrounds_dir=None, haven_path=None):
    """
    Unified function to process single objects or pairs of objects.
    Only saves background versions with enhanced lighting.
    """
    
    # Determine scene type and identifier
    if len(obj_paths) == 1:
        scene_type = "single"
        obj_filename = os.path.basename(obj_paths[0])
        obj_name = os.path.splitext(obj_filename)[0]
        scene_identifier = f"{obj_name}"
        print(f"\nProcessing single object: {obj_name}")
    else:
        scene_type = "pair"
        obj_names = [os.path.splitext(os.path.basename(path))[0] for path in obj_paths]
        scene_identifier = "_".join(obj_names)
        print(f"\nProcessing pair: {' + '.join(obj_names)}")
    
    # Load HDR files - FIXED: Pre-select HDRIs to prevent stall bug
    hdr_files = []
    selected_hdris = []
    if haven_path and os.path.exists(haven_path):
        try:
            hdr_files = get_hdr_img_paths_from_haven(haven_path)
            if hdr_files:
                # PRE-SELECT exactly num_images HDRIs to prevent repeat selection loops
                selected_hdris = [random.choice(hdr_files) for _ in range(num_images)]
                print(f"Pre-selected {len(selected_hdris)} HDRIs from {len(hdr_files)} available")
            else:
                print(f"Warning: No HDR files found in {haven_path}")
        except Exception as e:
            print(f"Warning: Could not load HDR files: {e}")
    
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
        
        for i, obj_path in enumerate(obj_paths):
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
            obj.set_cp("obj_identifier", obj_identifier)
            
            # Position objects side by side if pair
            if len(obj_paths) > 1:
                offset = [-2.0, 0, 0] if i == 0 else [2.0, 0, 0]
                current_location = obj.get_location()
                obj.set_location(current_location + np.array(offset))
            
            # Apply materials with full opacity
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
                
                # Set blend mode to opaque
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
            
            # Detect keypoints based on category
            if category == 'needle_holder':
                keypoints_3d = KeypointDetector.detect_needle_holder_keypoints(verts_world)
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
            
            # FIXED: Set HDRI background using pre-selected HDRI (no repeat selection)
            if selected_hdris and poses < len(selected_hdris):
                selected_hdr_file = selected_hdris[poses]
                bproc.world.set_world_background_hdr_img(
                    selected_hdr_file,
                    strength=1.0,
                    rotation_euler=[0.0, 0.0, random.uniform(0, 2 * np.pi)]
                )
                print(f"[HDRI] Frame {poses}: Using {os.path.basename(selected_hdr_file)}")
            
            # Focus on the leftmost tool for camera targeting
            focus_obj = min(loaded_objects, key=lambda d: d['obj'].get_location()[0])['obj']

            # Sample camera location with controlled parameters
            # Sample camera location with controlled parameters
            if scene_type == "single":
                # Position camera further back and higher for smaller tool appearance
                location = bproc.sampler.shell(
                    center=focus_obj.get_location(),
                    radius_min=6.0,  # Increased distance for smaller tool appearance
                    radius_max=8.0,  # Even further back
                    elevation_min=60,  # Higher elevation for better top-down view
                    elevation_max=80   # More overhead perspective
                )

                # Get tool's actual dimensions and center
                tool_center = focus_obj.get_location()
                tool_bbox = focus_obj.get_bound_box()

                # Calculate tool's approximate length (assuming tool is elongated along one axis)
                tool_dimensions = np.array(tool_bbox).max(axis=0) - np.array(tool_bbox).min(axis=0)
                tool_length = max(tool_dimensions)  # Get the longest dimension

                # Create lookat point that ensures tip points downward
                # Position the lookat point below the tool center to force downward orientation
                lookat_point = tool_center + np.array([
                    random.uniform(-0.2, 0.2),  # Small horizontal variation for natural look
                    random.uniform(-0.2, 0.2),  # Small Y variation
                    -tool_length * 0.4  # Point below tool center (tip area)
                ])

                # Alternative approach: If you know which end is the tip, use tool's local coordinates
                # This assumes the tool's tip is along the negative Z-axis in local coordinates
                tool_rotation = focus_obj.get_rotation_euler()

                # Create rotation matrix to ensure tool points downward
                # Force the tool's main axis to point toward negative Z (downward)
                desired_forward = np.array([0, 0, -1])  # Point straight down

                # Add some random variation for natural diagonal orientation
                diagonal_offset = np.array([
                    random.uniform(-0.3, 0.3),  # X variation
                    random.uniform(-0.3, 0.3),  # Y variation  
                    0  # Keep Z pointing down
                ])

                # Normalize the direction
                forward_direction = desired_forward + diagonal_offset
                forward_direction = forward_direction / np.linalg.norm(forward_direction)

                # Set camera rotation to look at the adjusted point
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    lookat_point - location, 
                    inplane_rot=np.deg2rad(np.random.uniform(-130, -90))  # Reduced rotation range for more controlled orientation
                )
            else:  # pair
                location = bproc.sampler.shell(
                    center=scene_center,
                    radius_min=3.2,
                    radius_max=3.5,
                    elevation_min=40,
                    elevation_max=55
                )
                lookat_point = scene_center
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    lookat_point - location,
                    inplane_rot = np.deg2rad(np.random.uniform(-130, -90))

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
            print(f"Warning: Only generated {poses}/{num_images} poses for {scene_identifier}")
        
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
        
        # Create output directories
        images_dir = os.path.join(output_dir, 'images_with_background')
        viz_dir = os.path.join(output_dir, 'visualizations_with_background')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each rendered frame - ONLY SAVE BACKGROUND VERSIONS
        processed_images = []
        
        for i, image in enumerate(data["colors"]):
            # Generate filenames
            if scene_type == "single":
                image_filename = f"{scene_identifier}_{i:06d}.png"
                viz_filename = f"vis_{scene_identifier}_{i:06d}.png"
            else:  # pair
                image_filename = f"pair_{scene_identifier}_{i:06d}.png"
                viz_filename = f"vis_pair_{scene_identifier}_{i:06d}.png"
            
            # Collect all keypoints and bboxes for this frame
            frame_objects = []
            
            for obj_idx, obj_data in enumerate(loaded_objects):
                obj_identifier = obj_data['identifier']
                category = obj_data['category']
                original_name = obj_data['original_name']
                
                # Get keypoints for this object and frame with view-dependent labeling
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
            
            # Apply background integration with enhanced lighting
            bg_image = enhance_background_integration(image, backgrounds_dir)
            
            # Save background image
            bg_image_path = os.path.join(images_dir, image_filename)
            plt.imsave(bg_image_path, bg_image)
            
            # Create visualization for background version
            if len(frame_objects) == 1:
                # Single object visualization
                obj_data = frame_objects[0]
                fig = create_visualization(
                    bg_image, obj_data["keypoints"], keypoint_colors, 
                    obj_data["object_id"], obj_data["category"], i, 
                    title_suffix=" (with Background)", bbox=obj_data["bbox"]
                )
            else:
                # Multi-object visualization
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
                title = f'Tool Pair - Frame {i} (with Background, {len(frame_objects)} objects)'
                ax.set_title(title, fontsize=10)
                plt.tight_layout()
            
            viz_path = os.path.join(viz_dir, viz_filename)
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store processed image info
            processed_images.append({
                "file_name": image_filename,
                "objects": frame_objects,
                "scene_type": scene_type,
                "num_objects": len(frame_objects)
            })
        
        print(f"Successfully processed {scene_identifier}: {poses} images rendered")
        return True, scene_identifier, poses, processed_images
        
    except Exception as e:
        print(f"Error processing {scene_identifier}: {e}")
        import traceback
        traceback.print_exc()
        return False, scene_identifier, 0, []

def main():
    parser = argparse.ArgumentParser(description='Generate surgical tool datasets with background integration only')
    parser.add_argument('--tools_dir', 
                        default="/datashare/project/surgical_tools_models", 
                        help="Path to surgical tools models directory")
    parser.add_argument('--camera_params', 
                        default="/datashare/project/camera.json", 
                        help="Camera intrinsics in json format")
    parser.add_argument('--output_dir', 
                        default="output_background_only", 
                        help="Output directory")
    parser.add_argument('--num_images', 
                        type=int, default=2, 
                        help="Number of images per object/pair")
    parser.add_argument('--categories',
                        nargs='+',
                        default=['needle_holder', 'tweezers'],
                        help="Categories to process")
    parser.add_argument('--backgrounds_dir',
                        default="/datashare/project/train2017",
                        help="Directory containing background images")
    parser.add_argument('--haven_path',
                        default="/datashare/project/haven/",
                        help="Path to the haven HDRI images")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.backgrounds_dir):
        print(f"Error: Background directory does not exist: {args.backgrounds_dir}")
        return
    
    if not os.path.exists(args.haven_path):
        print(f"Error: Haven path does not exist: {args.haven_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize annotations
    annotations = {"images": []}
    
    # Object counter for generating identifiers
    object_counter = {"count": 1}
    
    # Load camera parameters
    with open(args.camera_params, "r") as file:
        camera_params = json.load(file)
    
    # Find all object files organized by category
    objects_by_category = {}
    
    for category in args.categories:
        category_path = os.path.join(args.tools_dir, category)
        if os.path.exists(category_path):
            obj_files = glob.glob(os.path.join(category_path, "*.obj"))
            objects_by_category[category] = obj_files
            print(f"Found {len(obj_files)} objects in {category}")
        else:
            print(f"Warning: Category directory not found: {category_path}")
            objects_by_category[category] = []
    
    total_single_objects = sum(len(objs) for objs in objects_by_category.values())
    
    # Calculate pairs
    tweezers = objects_by_category.get('tweezers', [])
    needle_holders = objects_by_category.get('needle_holder', [])
    total_pairs = len(tweezers) * len(needle_holders)
    
    print(f"\nPlanned processing:")
    print(f"  Single objects: {total_single_objects} (each gets {args.num_images} images)")
    print(f"  Tool pairs: {total_pairs} (each gets {args.num_images} images)")
    print(f"  Total images: {(total_single_objects + total_pairs) * args.num_images}")
    print(f"  Output: Background-enhanced images only")
    
    # Initialize BlenderProc
    print("Initializing BlenderProc...")
    bproc.init()
    
    try:
        successful_scenes = []
        failed_scenes = []
        
        # Process single objects first
        print(f"\n{'='*60}")
        print("PROCESSING SINGLE OBJECTS")
        print(f"{'='*60}")
        
        scene_count = 0
        for category, obj_paths in objects_by_category.items():
            for obj_path in obj_paths:
                scene_count += 1
                obj_name = os.path.splitext(os.path.basename(obj_path))[0]
                print(f"\n[{scene_count}/{total_single_objects}] Processing: {obj_name}")
                
                success, scene_id, num_rendered, processed_images = process_objects(
                    [obj_path],
                    camera_params,
                    args.output_dir,
                    args.num_images,
                    object_counter=object_counter,
                    backgrounds_dir=args.backgrounds_dir,
                    haven_path=args.haven_path
                )
                
                if success:
                    successful_scenes.append((scene_id, num_rendered, "single"))
                    annotations["images"].extend(processed_images)
                else:
                    failed_scenes.append((scene_id, "single"))
        
        # Process tool pairs
        if total_pairs > 0:
            print(f"\n{'='*60}")
            print("PROCESSING TOOL PAIRS")
            print(f"{'='*60}")
            
            pair_count = 0
            for tweezers_path in tweezers:
                for needle_holder_path in needle_holders:
                    pair_count += 1
                    tw_name = os.path.splitext(os.path.basename(tweezers_path))[0]
                    nh_name = os.path.splitext(os.path.basename(needle_holder_path))[0]
                    print(f"\n[{pair_count}/{total_pairs}] Processing pair: {tw_name} + {nh_name}")
                    
                    success, scene_id, num_rendered, processed_images = process_objects(
                        [tweezers_path, needle_holder_path],
                        camera_params,
                        args.output_dir,
                        args.num_images,
                        object_counter=object_counter,
                        backgrounds_dir=args.backgrounds_dir,
                        haven_path=args.haven_path
                    )
                    
                    if success:
                        successful_scenes.append((scene_id, num_rendered, "pair"))
                        annotations["images"].extend(processed_images)
                    else:
                        failed_scenes.append((scene_id, "pair"))
        
        # Save annotations
        annotations_file = os.path.join(args.output_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"\nAnnotations saved to: {annotations_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed scenes: {len(successful_scenes)}")
        print(f"Failed scenes: {len(failed_scenes)}")
        
        # Count images by type
        single_images = len([img for img in annotations['images'] if img.get('scene_type') == 'single'])
        pair_images = len([img for img in annotations['images'] if img.get('scene_type') == 'pair'])
        total_images = len(annotations['images'])
        
        print(f"Images generated:")
        print(f"  Single objects: {single_images}")
        print(f"  Tool pairs: {pair_images}")
        print(f"  Total: {total_images}")
        
        print(f"\nOutput directories:")
        print(f"  Images: {os.path.join(args.output_dir, 'images_with_background')}")
        print(f"  Visualizations: {os.path.join(args.output_dir, 'visualizations_with_background')}")
        print(f"  Annotations: {annotations_file}")
        
        if successful_scenes:
            print("\nSuccessful scenes:")
            single_scenes = [s for s in successful_scenes if s[2] == 'single']
            pair_scenes = [s for s in successful_scenes if s[2] == 'pair']
            
            if single_scenes:
                print(f"  Single objects ({len(single_scenes)}):")
                for scene_id, num_rendered, scene_type in single_scenes:
                    print(f"    ✓ {scene_id}: {num_rendered} images")
            
            if pair_scenes:
                print(f"  Tool pairs ({len(pair_scenes)}):")
                for scene_id, num_rendered, scene_type in pair_scenes:
                    print(f"    ✓ {scene_id}: {num_rendered} images")
        
        if failed_scenes:
            print("\nFailed scenes:")
            for scene_id, scene_type in failed_scenes:
                print(f"  ✗ {scene_id} ({scene_type})")
        
        # Create comprehensive summary
        summary_file = os.path.join(args.output_dir, "processing_summary.json")
        summary_data = {
            "configuration": {
                "backgrounds_dir": args.backgrounds_dir,
                "haven_path": args.haven_path,
                "num_images_per_scene": args.num_images,
                "categories": args.categories,
                "output_type": "background_enhanced_only"
            },
            "results": {
                "total_scenes_planned": total_single_objects + total_pairs,
                "single_objects_planned": total_single_objects,
                "tool_pairs_planned": total_pairs,
                "successful_scenes": len(successful_scenes),
                "failed_scenes": len(failed_scenes),
                "total_images": total_images,
                "single_object_images": single_images,
                "tool_pair_images": pair_images
            },
            "output_structure": {
                "images": os.path.join(args.output_dir, 'images_with_background'),
                "visualizations": os.path.join(args.output_dir, 'visualizations_with_background'),
                "annotations": annotations_file
            },
            "successful_scenes": [
                {"identifier": scene_id, "images_rendered": num_rendered, "type": scene_type}
                for scene_id, num_rendered, scene_type in successful_scenes
            ],
            "failed_scenes": [
                {"identifier": scene_id, "type": scene_type}
                for scene_id, scene_type in failed_scenes
            ]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nComprehensive summary saved to: {summary_file}")
        
        # Final statistics
        if total_images > 0:
            success_rate = len(successful_scenes) / (total_single_objects + total_pairs) * 100
            print(f"\nFinal Statistics:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average images per successful scene: {total_images / len(successful_scenes):.1f}")
            print(f"  Background enhancement: Applied to all images")
            print(f"  HDRI lighting: Applied to all scenes")
    
    finally:
        # Clean up BlenderProc
        bproc.clean_up()

if __name__ == "__main__":
    main()