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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    """Category-specific keypoint detection (actual vertices only)"""

    @staticmethod
    def detect_needle_holder_keypoints(verts_world, ref_right_dir=None):
        """
        Needle holder strategy with actual vertex picks.
        Keys: top_left, top_right, bottom_left, bottom_right, middle_left, middle_right, joint_center
        """
        V = np.asarray(verts_world, dtype=float)
        X, Y, Z = V[:, 0], V[:, 1], V[:, 2]

        kps = {}

        # --- Tips (top 15% Z) ---
        z_top = np.percentile(Z, 85)
        tip_idx = np.where(Z >= z_top)[0]
        if tip_idx.size == 0:
            z_top = Z.max()
            tip_idx = np.where(Z >= z_top - 1e-8)[0]
        kps['top_left']  = V[tip_idx[np.argmin(X[tip_idx])]]
        kps['top_right'] = V[tip_idx[np.argmax(X[tip_idx])]]

        # --- Butt (bottom 15% Z) ---
        z_bot = np.percentile(Z, 15)
        butt_idx = np.where(Z <= z_bot)[0]
        if butt_idx.size == 0:
            z_bot = Z.min()
            butt_idx = np.where(Z <= z_bot + 1e-8)[0]
        kps['bottom_left']  = V[butt_idx[np.argmin(X[butt_idx])]]
        kps['bottom_right'] = V[butt_idx[np.argmax(X[butt_idx])]]

        # --- Mid band (30–70% Z) ---
        z_lo, z_hi = np.percentile(Z, 30), np.percentile(Z, 70)
        mid_band = np.where((Z >= z_lo) & (Z <= z_hi))[0]

        def nearest_vertex(idx_subset, target):
            pts = V[idx_subset]
            return idx_subset[np.argmin(np.einsum('ij,ij->i', pts - target, pts - target))]

        got_left = got_right = False
        if mid_band.size > 0:
            X_band = X[mid_band]
            x_q1, x_q3 = np.percentile(X_band, 25), np.percentile(X_band, 75)
            left_idx = mid_band[np.where(X_band <= x_q1)[0]]
            right_idx = mid_band[np.where(X_band >= x_q3)[0]]

            if left_idx.size > 0:
                left_centroid = V[left_idx].mean(axis=0)
                kps['middle_left'] = V[nearest_vertex(left_idx, left_centroid)]
                got_left = True
            if right_idx.size > 0:
                right_centroid = V[right_idx].mean(axis=0)
                kps['middle_right'] = V[nearest_vertex(right_idx, right_centroid)]
                got_right = True

        if not (got_left and got_right):
            x_med = np.median(X)
            left_idx = np.where(X <= x_med)[0]
            right_idx = np.where(X >= x_med)[0]
            if not got_left and left_idx.size > 0:
                left_centroid = V[left_idx].mean(axis=0)
                kps['middle_left'] = V[nearest_vertex(left_idx, left_centroid)]
            if not got_right and right_idx.size > 0:
                right_centroid = V[right_idx].mean(axis=0)
                kps['middle_right'] = V[nearest_vertex(right_idx, right_centroid)]

        if 'middle_left' not in kps:
            kps['middle_left']  = V[[np.argmin(np.abs(X - np.median(X)))]][0]
        if 'middle_right' not in kps:
            kps['middle_right'] = V[[np.argmax(np.abs(X - np.median(X)))]][0]

        # --- Joint center ---
        tips_center  = 0.5 * (kps['top_left'] + kps['top_right'])
        shaft_center = 0.5 * (kps['middle_left'] + kps['middle_right'])
        virtual_joint = tips_center + 0.3 * (shaft_center - tips_center)
        j_idx = np.argmin(np.einsum('ij,ij->i', V - virtual_joint, V - virtual_joint))
        kps['joint_center'] = V[j_idx]

        return kps

    @staticmethod
    def detect_tweezers_keypoints(verts_world, ref_right_dir=None):
        """
        Tweezers strategy with actual vertex picks.
        Keys: bottom_tip, top_left, top_right, mid_left, mid_right
        """
        V = np.asarray(verts_world, dtype=float)
        X, Y, Z = V[:, 0], V[:, 1], V[:, 2]

        kps = {}

        # --- Bottom tip: min Z vertex ---
        kps['bottom_tip'] = V[np.argmin(Z)]

        # Split by median X into arms
        x_med = np.median(X)
        left_arm  = np.where(X <= x_med)[0]
        right_arm = np.where(X >= x_med)[0]
        if left_arm.size == 0 or right_arm.size == 0:
            left_arm  = np.where(X <= np.percentile(X, 50))[0]
            right_arm = np.where(X >= np.percentile(X, 50))[0]

        # Rear tips: top 20% Z per arm, extreme X
        def arm_top_extreme(arm_idx, choose_min_x=True):
            if arm_idx.size == 0:
                return None
            z_thr = np.percentile(Z[arm_idx], 80)
            cand = arm_idx[np.where(Z[arm_idx] >= z_thr)[0]]
            if cand.size == 0:
                zmax = Z[arm_idx].max()
                cand = arm_idx[np.where(Z[arm_idx] >= zmax - 1e-8)[0]]
            if cand.size == 0:
                return None
            return V[cand[np.argmin(X[cand]) if choose_min_x else np.argmax(X[cand])]]

        kps['top_left']  = arm_top_extreme(left_arm,  True)
        kps['top_right'] = arm_top_extreme(right_arm, False)

        # Midpoints per arm: closest to arm-median Z
        def arm_mid(arm_idx):
            if arm_idx.size == 0:
                return None
            z_med = np.median(Z[arm_idx])
            return V[arm_idx[np.argmin(np.abs(Z[arm_idx] - z_med))]]

        kps['mid_left']  = arm_mid(left_arm)
        kps['mid_right'] = arm_mid(right_arm)

        # Fallbacks
        if kps['top_left'] is None:
            z_thr = np.percentile(Z, 80); cand = np.where(Z >= z_thr)[0]
            if cand.size == 0: cand = np.where(Z >= Z.max() - 1e-8)[0]
            kps['top_left'] = V[cand[np.argmin(X[cand])]]
        if kps['top_right'] is None:
            z_thr = np.percentile(Z, 80); cand = np.where(Z >= z_thr)[0]
            if cand.size == 0: cand = np.where(Z >= Z.max() - 1e-8)[0]
            kps['top_right'] = V[cand[np.argmax(X[cand])]]
        if kps['mid_left'] is None:
            kps['mid_left'] = V[np.argmin(np.abs(Z - np.median(Z[X <= x_med])))]
        if kps['mid_right'] is None:
            kps['mid_right'] = V[np.argmin(np.abs(Z - np.median(Z[X >= x_med])))]
        return kps

    @staticmethod
    def assign_view_dependent_labels(keypoints_3d, keypoints_2d_projected, frame_idx):
        """
        After projection, enforce left/right in the image:
        If 'left' 2D x > 'right' 2D x, swap for that pair.
        """
        if not keypoints_2d_projected or len(keypoints_2d_projected) == 0:
            return keypoints_2d_projected

        kp_names = list(keypoints_3d.keys())
        kp_2d = {}
        for i, name in enumerate(kp_names):
            pt = keypoints_2d_projected[i] if i < len(keypoints_2d_projected) else None
            kp_2d[name] = None if pt is None else [float(pt[0]), float(pt[1])]

        pairs = [
            ("bottom_left", "bottom_right"),
            ("top_left", "top_right"),
            ("middle_left", "middle_right"),
            ("mid_left", "mid_right"),
        ]
        for L, R in pairs:
            if L in kp_2d and R in kp_2d and kp_2d[L] is not None and kp_2d[R] is not None:
                if kp_2d[L][0] > kp_2d[R][0]:
                    kp_2d[L], kp_2d[R] = kp_2d[R], kp_2d[L]

        return kp_2d

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
                if z_preference == 'min' and x_preference == 'min':
                    scores = z_coords + x_coords
                    return verts_array[np.argmin(scores)]
                elif z_preference == 'min' and x_preference == 'max':
                    scores = z_coords - x_coords
                    return verts_array[np.argmin(scores)]
                elif z_preference == 'max' and x_preference == 'min':
                    scores = z_coords - x_coords
                    return verts_array[np.argmax(scores)]
                else:
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
        return {
            'bottom_left': 'red',
            'bottom_right': 'blue',
            'top_left': 'green',
            'top_right': 'orange',
            'joint_center': 'purple',
            'middle_left': 'cyan',
            'middle_right': 'magenta',
            'bottom_tip': 'yellow',
            'mid_left': 'pink',
            'mid_right': 'brown',
        }

def get_hdr_img_paths_from_haven(data_path: str) -> list:
    if os.path.exists(data_path):
        data_path = os.path.join(data_path, "hdris")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The folder: {data_path} does not contain a folder name hdris. Please use the download script.")
    else:
        raise FileNotFoundError(f"The data path does not exists: {data_path}")

    hdr_files = glob.glob(os.path.join(data_path, "*", "*.hdr"))
    hdr_files.sort()
    return hdr_files

def get_background_files(backgrounds_dir: str) -> list:
    background_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        background_files.extend(glob.glob(os.path.join(backgrounds_dir, f"*.{ext}")))
        background_files.extend(glob.glob(os.path.join(backgrounds_dir, f"*.{ext.upper()}")))
    return background_files

def safe_set_principled_shader_value(material, input_name, value):
    try:
        material.set_principled_shader_value(input_name, value)
        return True
    except KeyError:
        print(f"Warning: Shader input '{input_name}' not available, skipping...")
        return False

def create_visualization(image, keypoints_2d, keypoint_colors, obj_identifier, category, frame_idx, title_suffix="", bbox=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    # --- draw connection lines based on tool type ---
    if category == 'needle_holder':
        connections = [
            ("bottom_left", "middle_left"),
            ("middle_left", "joint_center"),
            ("joint_center", "top_left"),
            ("bottom_right", "middle_right"),
            ("middle_right", "joint_center"),
            ("joint_center", "top_right"),
        ]
    elif category == 'tweezers':
        connections = [
            ("top_left",  "mid_left"),
            ("mid_left",  "bottom_tip"),
            ("top_right", "mid_right"),
            ("mid_right", "bottom_tip"),
        ]
    else:
        connections = []

    # plot lines first so points render on top
    for a, b in connections:
        pa = keypoints_2d.get(a); pb = keypoints_2d.get(b)
        if pa is not None and pb is not None:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    '-', linewidth=2,
                    color=keypoint_colors.get(a, 'white'),
                    alpha=0.9, zorder=1)

    # --- draw keypoints ---
    legend_patches = []
    for kp_name, kp_color in keypoint_colors.items():
        if kp_name in keypoints_2d:
            pt = keypoints_2d[kp_name]
            if pt is not None:
                x, y = pt
                ax.plot(x, y, 'o', markersize=6, color=kp_color, markeredgecolor='white', markeredgewidth=1, zorder=2)
                legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))

    # --- draw bbox (unchanged) ---
    if bbox is not None and len(bbox) == 4:
        rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 5, 'bbox', color='lime', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)

    ax.axis('off')
    title = f'{category.replace("_", " ").title()}: {obj_identifier} - Frame {frame_idx}{title_suffix}'
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig


def process_objects(obj_paths, camera_params, output_dir, num_images=25,
                   object_counter=None, backgrounds_dir=None, haven_path=None, global_hdri_counter=None):
    """
    Unified function to process single objects or pairs of objects.
    Only saves background versions with enhanced lighting.
    """
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

    hdr_files = []
    if haven_path and os.path.exists(haven_path):
        try:
            hdr_files = get_hdr_img_paths_from_haven(haven_path)
            if not hdr_files:
                print(f"Warning: No HDR files found in {haven_path}")
        except Exception as e:
            print(f"Warning: Could not load HDR files: {e}")

    try:
        bproc.clean_up(clean_up_camera=False)

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

        loaded_objects = []
        all_keypoints_3d = {}

        # --- Load objects, position, (tiny spacing), materials ---
        for i, obj_path in enumerate(obj_paths):
            obj_filename = os.path.basename(obj_path)
            obj_name = os.path.splitext(obj_filename)[0]

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

            obj_identifier = f"{obj_name}"
            object_counter['count'] += 1

            obj = bproc.loader.load_obj(obj_path)[0]
            obj.set_cp("category_id", category_id)
            obj.set_cp("obj_identifier", obj_identifier)

            if len(obj_paths) > 1:
                # tighter spacing for pair
                offset_mag = random.uniform(1.3, 1.6)
                offset = [-offset_mag, 0, 0] if i == 0 else [offset_mag, 0, 0]
                obj.set_location(obj.get_location() + np.array(offset))
                # (we'll steer yaw toward a common target *after* both are placed)

            materials = obj.get_materials()
            if len(materials) > 0:
                mat = materials[0]
                safe_set_principled_shader_value(mat, "IOR", random.uniform(1.0, 2.5))
                safe_set_principled_shader_value(mat, "Roughness", random.uniform(0.1, 0.8))
                safe_set_principled_shader_value(mat, "Metallic", 1)
                safe_set_principled_shader_value(mat, "Alpha", 1.0)
                safe_set_principled_shader_value(mat, "Transmission", 0.0)
                try:
                    if hasattr(mat.blender_obj, 'blend_method'):
                        mat.blender_obj.blend_method = 'OPAQUE'
                    elif hasattr(mat.blender_obj, 'use_transparency'):
                        mat.blender_obj.use_transparency = False
                except Exception as e:
                    print(f"Warning: Could not set blend mode: {e}")
                if len(materials) > 1:
                    mat2 = materials[1]
                    random_gold_hsv_color = np.random.uniform([0.03, 0.95, 0.8], [0.25, 1.0, 1.0])
                    random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0]
                    safe_set_principled_shader_value(mat2, "Base Color", random_gold_color)
                    safe_set_principled_shader_value(mat2, "IOR", random.uniform(1.0, 2.5))
                    safe_set_principled_shader_value(mat2, "Roughness", random.uniform(0.1, 0.8))
                    safe_set_principled_shader_value(mat2, "Metallic", 1)
                    safe_set_principled_shader_value(mat2, "Alpha", 1.0)
                    safe_set_principled_shader_value(mat2, "Transmission", 0.0)
                    try:
                        if hasattr(mat2.blender_obj, 'blend_method'):
                            mat2.blender_obj.blend_method = 'OPAQUE'
                        elif hasattr(mat2.blender_obj, 'use_transparency'):
                            mat2.blender_obj.use_transparency = False
                    except Exception as e:
                        print(f"Warning: Could not set blend mode for material 2: {e}")

            # Initial verts (will recompute after “aim to center”)
            mesh = obj.get_mesh()
            verts = mesh.vertices
            obj2world = obj.get_local2world_mat()
            verts_world = [obj2world @ np.append(v.co, 1.0) for v in verts]
            verts_world = [v[:3] for v in verts_world]

            loaded_objects.append({
                'obj': obj,
                'identifier': obj_identifier,
                'category': category,
                'original_name': obj_name,
                'verts_world': verts_world
            })

        # --- Compute scene center, aim both tools toward a common noisy target (pair only) ---
        if len(loaded_objects) > 1:
            all_locations = [d['obj'].get_location() for d in loaded_objects]
            scene_center = np.mean(all_locations, axis=0)

            # target near scene_center in XY (±5–15% noise of typical offset ~1.5)
            jitter = random.uniform(0.05, 0.15) * 10.0  # small world jitter
            target_xy = scene_center[:2] + np.array([random.uniform(-jitter, jitter),
                                                     random.uniform(-jitter, jitter)])

            for d in loaded_objects:
                obj = d['obj']
                loc_xy = obj.get_location()[:2]
                bearing = np.arctan2(target_xy[1] - loc_xy[1], target_xy[0] - loc_xy[0])
                eul = obj.get_rotation_euler()
                # apply a small bounded delta toward bearing (keeps natural variation)
                curr_yaw = float(eul[2])
                # normalize smallest angle difference
                diff = (bearing - curr_yaw + np.pi) % (2*np.pi) - np.pi
                # limit to ~±15°–18°
                limit = np.deg2rad(random.uniform(10.0, 18.0))
                diff = max(-limit, min(limit, diff))
                eul[2] = curr_yaw + diff
                obj.set_rotation_euler(eul)

            # Recompute verts & keypoints after the yaw adjustment
            all_keypoints_3d.clear()
            for d in loaded_objects:
                obj = d['obj']
                mesh = obj.get_mesh()
                verts = mesh.vertices
                obj2world = obj.get_local2world_mat()
                verts_world = [obj2world @ np.append(v.co, 1.0) for v in verts]
                verts_world = [v[:3] for v in verts_world]
                d['verts_world'] = verts_world

                if d['category'] == 'needle_holder':
                    keypoints_3d = KeypointDetector.detect_needle_holder_keypoints(verts_world)
                elif d['category'] == 'tweezers':
                    keypoints_3d = KeypointDetector.detect_tweezers_keypoints(verts_world)
                else:
                    verts_array = np.array(verts_world)
                    keypoints_3d = KeypointDetector._get_standard_corners_actual_verts(
                        verts_array, verts_array[:, 0], verts_array[:, 1], verts_array[:, 2]
                    )
                all_keypoints_3d[d['identifier']] = keypoints_3d
        else:
            # single: compute keypoints once
            d = loaded_objects[0]
            if d['category'] == 'needle_holder':
                all_keypoints_3d[d['identifier']] = KeypointDetector.detect_needle_holder_keypoints(d['verts_world'])
            elif d['category'] == 'tweezers':
                all_keypoints_3d[d['identifier']] = KeypointDetector.detect_tweezers_keypoints(d['verts_world'])
            else:
                verts_array = np.array(d['verts_world'])
                all_keypoints_3d[d['identifier']] = KeypointDetector._get_standard_corners_actual_verts(
                    verts_array, verts_array[:, 0], verts_array[:, 1], verts_array[:, 2]
                )

        # --- Light and camera sampling ---
        light = bproc.types.Light()
        light.set_type("POINT")

        if len(loaded_objects) > 1:
            all_locations = [obj_data['obj'].get_location() for obj_data in loaded_objects]
            scene_center = np.mean(all_locations, axis=0)
        else:
            scene_center = loaded_objects[0]['obj'].get_location()

        light.set_location(bproc.sampler.shell(
            center=scene_center, radius_min=1, radius_max=5, elevation_min=1, elevation_max=89
        ))
        light.set_energy(random.uniform(100, 1000))

        fx = camera_params["fx"]; fy = camera_params["fy"]
        cx = camera_params["cx"]; cy = camera_params["cy"]
        im_width = camera_params["width"]; im_height = camera_params["height"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height)

        bpy.context.scene.frame_set(0)

        all_projected_keypoints = {obj_id: {k: [] for k in all_keypoints_3d[obj_id].keys()} for obj_id in all_keypoints_3d}

        poses = 0
        tries = 0
        max_tries = 10000

        while tries < max_tries and poses < num_images:
            world = bpy.context.scene.world
            if world and world.node_tree and "Background" in world.node_tree.nodes:
                world.node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)

            if hdr_files and global_hdri_counter is not None:
                selected_hdr_file = hdr_files[global_hdri_counter['count'] % len(hdr_files)]
                global_hdri_counter['count'] += 1
                bproc.world.set_world_background_hdr_img(
                    selected_hdr_file, strength=1.0, rotation_euler=[0.0, 0.0, random.uniform(0, 2 * np.pi)]
                )

            focus_obj = min(loaded_objects, key=lambda d: d['obj'].get_location()[0])['obj']

            if scene_type == "single":
                location = bproc.sampler.shell(
                    center=focus_obj.get_location(), radius_min=6.0, radius_max=8.0, elevation_min=60, elevation_max=80
                )
                tool_center = focus_obj.get_location()
                tool_bbox = focus_obj.get_bound_box()
                tool_dimensions = np.array(tool_bbox).max(axis=0) - np.array(tool_bbox).min(axis=0)
                tool_length = max(tool_dimensions)
                lookat_point = tool_center + np.array([
                    random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), -tool_length * 0.4
                ])
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    lookat_point - location, inplane_rot=np.deg2rad(np.random.uniform(-130, -90))
                )
            else:
                # Top-down, smaller scale, tools vertical-ish
                location = bproc.sampler.shell(
                    center=scene_center, radius_min=6.0, radius_max=8.0, elevation_min=70, elevation_max=82
                )
                lookat_point = scene_center
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    lookat_point - location, inplane_rot=np.deg2rad(np.random.uniform(-105, -90))
                )

            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

            all_visible = all(obj_data['obj'] in bproc.camera.visible_objects(cam2world_matrix)
                              for obj_data in loaded_objects)

            if all_visible:
                bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
                if hdr_files:
                    print(f"[HDRI] Frame {poses}: Using {os.path.basename(selected_hdr_file)}")

                for obj_identifier, keypoints_3d in all_keypoints_3d.items():
                    kp_names = list(keypoints_3d.keys())
                    keypoint_coords_3d = np.array([keypoints_3d[k] for k in kp_names])
                    projected_2d = project_points(keypoint_coords_3d, frame=poses)
                    if projected_2d is not None and len(projected_2d) == len(kp_names):
                        for i, kp_name in enumerate(kp_names):
                            all_projected_keypoints[obj_identifier][kp_name].append(projected_2d[i])
                    else:
                        for kp_name in kp_names:
                            all_projected_keypoints[obj_identifier][kp_name].append(None)

                poses += 1

            tries += 1

        if poses < num_images:
            print(f"Warning: Only generated {poses}/{num_images} poses for {scene_identifier}")

        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_output_format(enable_transparency=True)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": 0})

        print(f"Rendering {poses} images...")
        data = bproc.renderer.render()

        keypoint_colors = KeypointDetector.get_keypoint_colors()

        images_dir = os.path.join(output_dir, 'images_with_background')
        viz_dir = os.path.join(output_dir, 'visualizations_with_background')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        processed_images = []

        for i, image in enumerate(data["colors"]):
            if scene_type == "single":
                image_filename = f"{scene_identifier}_{i:06d}.png"
                viz_filename = f"vis_{scene_identifier}_{i:06d}.png"
            else:
                image_filename = f"pair_{scene_identifier}_{i:06d}.png"
                viz_filename = f"vis_pair_{scene_identifier}_{i:06d}.png"

            frame_objects = []

            for obj_idx, obj_data in enumerate(loaded_objects):
                obj_identifier = obj_data['identifier']
                category = obj_data['category']
                original_name = obj_data['original_name']

                frame_keypoints_raw = []
                for kp_name in all_keypoints_3d[obj_identifier].keys():
                    if i < len(all_projected_keypoints[obj_identifier][kp_name]):
                        pt = all_projected_keypoints[obj_identifier][kp_name][i]
                        frame_keypoints_raw.append(pt)
                    else:
                        frame_keypoints_raw.append(None)

                frame_keypoints = KeypointDetector.assign_view_dependent_labels(
                    all_keypoints_3d[obj_identifier], frame_keypoints_raw, i
                )

                # --- bbox + mask ---
                obj_mask = None
                try:
                    instance_map = data["instance_segmaps"][i]
                    unique_instances = np.unique(instance_map)
                    valid_instances = [inst for inst in unique_instances if inst > 0]
                    if len(valid_instances) > obj_idx:
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

                # --- 2D correction: bbox clamp + anatomy-guided (minimal) ---
                if bbox[2] > 0 and bbox[3] > 0:
                    x0, y0 = bbox[0], bbox[1]
                    x1, y1 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                    xi0, yi0 = int(np.floor(x0)), int(np.floor(y0))
                    xi1, yi1 = int(np.ceil(x1)), int(np.ceil(y1))

                    mask_points = None
                    if obj_mask is not None and obj_mask.any():
                        sub = obj_mask[max(0, yi0):min(obj_mask.shape[0], yi1+1),
                                       max(0, xi0):min(obj_mask.shape[1], xi1+1)]
                        ys, xs = np.where(sub > 0)
                        if ys.size > 0:
                            mask_points = np.stack([xs + max(0, xi0), ys + max(0, yi0)], axis=1)

                    def clamp_to_bbox_xy(x, y):
                        cx = min(max(x, x0), x1)
                        cy = min(max(y, y0), y1)
                        return cx, cy

                    def proj_to_segment(p, a, b):
                        if a is None or b is None:
                            return p
                        ax, ay = a; bx, by = b
                        abx, aby = bx - ax, by - ay
                        denom = abx*abx + aby*aby
                        if denom < 1e-6:
                            return a
                        t = ((p[0]-ax)*abx + (p[1]-ay)*aby) / denom
                        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                        return [ax + t*abx, ay + t*aby]

                    for name, pt in list(frame_keypoints.items()):
                        if pt is None:
                            continue

                        x, y = float(pt[0]), float(pt[1])

                        # Tweezers bottom_tip: ensure inside bbox using ordered Z candidates, else clamp
                        if category == 'tweezers' and name == 'bottom_tip':
                            outside_now = (x < x0) or (x > x1) or (y < y0) or (y > y1)
                            if outside_now:
                                V = np.array(obj_data['verts_world'], dtype=float)
                                order = np.argsort(V[:, 2])  # increasing Z (bottom-most first)
                                found_inside = False
                                for idx in order:
                                    cand2d = project_points(V[idx:idx+1], frame=i)
                                    if cand2d is None or len(cand2d) == 0:
                                        continue
                                    cx, cy = float(cand2d[0][0]), float(cand2d[0][1])
                                    if (x0 <= cx <= x1) and (y0 <= cy <= y1):
                                        x, y = cx, cy
                                        found_inside = True
                                        break
                                if not found_inside:
                                    x, y = clamp_to_bbox_xy(x, y)

                        # Step 1: clamp outside -> edge
                        if (x < x0) or (x > x1) or (y < y0) or (y > y1):
                            x, y = clamp_to_bbox_xy(x, y)

                        # Step 2: inside but off-mask -> gentle constraint (skip for tweezers bottom_tip)
                        on_mask = False
                        if obj_mask is not None and obj_mask.any():
                            xi = int(round(x)); yi = int(round(y))
                            xi = max(0, min(obj_mask.shape[1]-1, xi))
                            yi = max(0, min(obj_mask.shape[0]-1, yi))
                            on_mask = obj_mask[yi, xi] > 0

                        if (not on_mask) and (obj_mask is not None and obj_mask.any()):
                            if category == 'needle_holder':
                                if 'left' in name:
                                    a = frame_keypoints.get('bottom_left'); b = frame_keypoints.get('top_left')
                                    if a is not None and b is not None: x, y = proj_to_segment([x, y], a, b)
                                elif 'right' in name:
                                    a = frame_keypoints.get('bottom_right'); b = frame_keypoints.get('top_right')
                                    if a is not None and b is not None: x, y = proj_to_segment([x, y], a, b)
                                elif name == 'joint_center':
                                    tl = frame_keypoints.get('top_left'); tr = frame_keypoints.get('top_right')
                                    ml = frame_keypoints.get('middle_left'); mr = frame_keypoints.get('middle_right')
                                    if tl and tr and ml and mr:
                                        a = [(tl[0]+tr[0])/2.0, (tl[1]+tr[1])/2.0]
                                        b = [(ml[0]+mr[0])/2.0, (ml[1]+mr[1])/2.0]
                                        x, y = proj_to_segment([x, y], a, b)

                            elif category == 'tweezers':
                                if name == 'bottom_tip':
                                    # keep as the chosen vertex projection; no midline override here
                                    pass
                                elif 'left' in name:
                                    a = frame_keypoints.get('mid_left');  b = frame_keypoints.get('top_left')
                                    if a is not None and b is not None: x, y = proj_to_segment([x, y], a, b)
                                elif 'right' in name:
                                    a = frame_keypoints.get('mid_right'); b = frame_keypoints.get('top_right')
                                    if a is not None and b is not None: x, y = proj_to_segment([x, y], a, b)

                            # If still off-mask, snap to nearest foreground ONLY if not bottom_tip
                            if not (category == 'tweezers' and name == 'bottom_tip'):
                                xi = int(round(x)); yi = int(round(y))
                                xi = max(0, min(obj_mask.shape[1]-1, xi))
                                yi = max(0, min(obj_mask.shape[0]-1, yi))
                                if obj_mask[yi, xi] == 0 and (mask_points is not None and mask_points.size > 0):
                                    dx = mask_points[:, 0] - x
                                    dy = mask_points[:, 1] - y
                                    j = int(np.argmin(dx*dx + dy*dy))
                                    x = float(mask_points[j, 0]); y = float(mask_points[j, 1])

                        # final clamp
                        x, y = clamp_to_bbox_xy(x, y)
                        frame_keypoints[name] = [float(x), float(y)]

                    # Re-enforce 2D left/right order after corrections
                    pairs = [
                        ("bottom_left", "bottom_right"),
                        ("top_left", "top_right"),
                        ("middle_left", "middle_right"),
                        ("mid_left", "mid_right"),
                    ]
                    for L, R in pairs:
                        if L in frame_keypoints and R in frame_keypoints:
                            lpt, rpt = frame_keypoints[L], frame_keypoints[R]
                            if lpt is not None and rpt is not None and lpt[0] > rpt[0]:
                                frame_keypoints[L], frame_keypoints[R] = frame_keypoints[R], frame_keypoints[L]

                frame_objects.append({
                    "object_id": obj_identifier,
                    "category": category,
                    "original_name": original_name,
                    "keypoints": frame_keypoints,
                    "bbox": bbox
                })

            bg_image = enhance_background_integration(image, backgrounds_dir)

            bg_image_path = os.path.join(images_dir, image_filename)
            plt.imsave(bg_image_path, bg_image)

            if len(frame_objects) == 1:
                obj_d = frame_objects[0]
                fig = create_visualization(
                    bg_image, obj_d["keypoints"], keypoint_colors,
                    obj_d["object_id"], obj_d["category"], i,
                    title_suffix=" (with Background)", bbox=obj_d["bbox"]
                )
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(bg_image)
                legend_patches = []
                for obj_d in frame_objects:
                    keypoints_2d = obj_d["keypoints"]
                    bbox = obj_d["bbox"]
                    for kp_name, kp_color in keypoint_colors.items():
                        if kp_name in keypoints_2d:
                            pt = keypoints_2d[kp_name]
                            if pt is not None:
                                x, y = pt
                                ax.plot(x, y, 'o', markersize=6, color=kp_color, markeredgecolor='white', markeredgewidth=1)
                                if not any(p.get_label() == kp_name for p in legend_patches):
                                    legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))
                    if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1] - 5, f'{obj_d["object_id"]}', color='lime', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)
                ax.axis('off')
                title = f'Tool Pair - Frame {i} (with Background, {len(frame_objects)} objects)'
                ax.set_title(title, fontsize=10)
                plt.tight_layout()

            viz_path = os.path.join(viz_dir, viz_filename)
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

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
    parser.add_argument('--tools_dir', default="/datashare/project/surgical_tools_models", help="Path to surgical tools models directory")
    parser.add_argument('--camera_params', default="/datashare/project/camera.json", help="Camera intrinsics in json format")
    parser.add_argument('--output_dir', default="NEW_VERSION_DATASET_FULL_09_08", help="Output directory")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images per object/pair")
    parser.add_argument('--categories', nargs='+', default=['needle_holder', 'tweezers'], help="Categories to process")
    parser.add_argument('--backgrounds_dir', default="/datashare/project/train2017", help="Directory containing background images")
    parser.add_argument('--haven_path', default="/datashare/project/haven/", help="Path to the haven HDRI images")
    args = parser.parse_args()

    if not os.path.exists(args.backgrounds_dir):
        print(f"Error: Background directory does not exist: {args.backgrounds_dir}")
        return
    if not os.path.exists(args.haven_path):
        print(f"Error: Haven path does not exist: {args.haven_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    annotations = {"images": []}
    object_counter = {"count": 1}
    global_hdri_counter = {"count": 0}

    with open(args.camera_params, "r") as file:
        camera_params = json.load(file)

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

    tweezers = objects_by_category.get('tweezers', [])
    needle_holders = objects_by_category.get('needle_holder', [])
    total_pairs = len(tweezers) * len(needle_holders)

    print(f"\nPlanned processing:")
    print(f"  Single objects: {total_single_objects} (each gets {args.num_images} images)")
    print(f"  Tool pairs: {total_pairs} (each gets {args.num_images} images)")
    print(f"  Total images: {(total_single_objects + total_pairs) * args.num_images}")
    print(f"  Output: Background-enhanced images only")

    print("Initializing BlenderProc...")
    bproc.init()

    try:
        successful_scenes = []
        failed_scenes = []

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
                    [obj_path], camera_params, args.output_dir, args.num_images,
                    object_counter=object_counter, backgrounds_dir=args.backgrounds_dir,
                    haven_path=args.haven_path, global_hdri_counter=global_hdri_counter
                )

                if success:
                    successful_scenes.append((scene_id, num_rendered, "single"))
                    annotations["images"].extend(processed_images)
                else:
                    failed_scenes.append((scene_id, "single"))

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
                        [tweezers_path, needle_holder_path], camera_params, args.output_dir, args.num_images,
                        object_counter=object_counter, backgrounds_dir=args.backgrounds_dir,
                        haven_path=args.haven_path, global_hdri_counter=global_hdri_counter
                    )

                    if success:
                        successful_scenes.append((scene_id, num_rendered, "pair"))
                        annotations["images"].extend(processed_images)
                    else:
                        failed_scenes.append((scene_id, "pair"))

        annotations_file = os.path.join(args.output_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"\nAnnotations saved to: {annotations_file}")

        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed scenes: {len(successful_scenes)}")
        print(f"Failed scenes: {len(failed_scenes)}")

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

        if total_images > 0:
            success_rate = len(successful_scenes) / (total_single_objects + total_pairs) * 100
            print(f"\nFinal Statistics:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average images per successful scene: {total_images / len(successful_scenes):.1f}")
            print(f"  Background enhancement: Applied to all images")
            print(f"  HDRI lighting: Applied to all scenes")

    finally:
        bproc.clean_up()

if __name__ == "__main__":
    main()
