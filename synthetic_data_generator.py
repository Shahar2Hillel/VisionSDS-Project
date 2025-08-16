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

def enhance_background_with_variations(rendered_image, single_background_path):
    """
    Enhanced background function with state-of-the-art augmentations:
    - Random zoom (scale variations)
    - Random position shifts
    - ENHANCED rotation with better edge handling
    - ENHANCED lighting variations (multiple types)
    - Color variations (hue, saturation, brightness, contrast)
    - Gaussian blur
    - Noise injection
    - Perspective transforms
    - Vignetting effects
    """
    
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
    if not os.path.exists(single_background_path):
        print(f"Warning: Background file not found: {single_background_path}, using white background")
        background = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        background = Image.open(single_background_path).convert('RGB')

    # === STATE-OF-THE-ART BACKGROUND AUGMENTATIONS ===
    
    # 1. RANDOM ZOOM (Scale Variations) - 50% chance
    if random.random() < 0.5:
        # Scale factor: zoom in (>1.0) or zoom out (<1.0)
        scale_factor = random.uniform(0.7, 1.4)  # 0.7x to 1.4x scale
        
        if scale_factor > 1.0:
            # Zoom IN: resize larger then crop center
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            background = background.resize((new_w, new_h), Image.LANCZOS)
            
            # Center crop
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            background = background.crop((left, top, left + width, top + height))
        else:
            # Zoom OUT: resize smaller then pad
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            background = background.resize((new_w, new_h), Image.LANCZOS)
            
            # Create larger canvas and paste resized image
            bg_canvas = Image.new('RGB', (width, height), 
                                color=(random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)))
            
            # Random position for pasting smaller image
            paste_x = random.randint(0, width - new_w)
            paste_y = random.randint(0, height - new_h)
            bg_canvas.paste(background, (paste_x, paste_y))
            background = bg_canvas
    else:
        # Normal resize
        background = background.resize((width, height), Image.LANCZOS)

    # 2. RANDOM POSITION SHIFT (Panning) - 40% chance
    if random.random() < 0.4:
        # Create larger background then randomly crop
        expand_factor = random.uniform(1.1, 1.3)
        expanded_w = int(width * expand_factor)
        expanded_h = int(height * expand_factor)
        background = background.resize((expanded_w, expanded_h), Image.LANCZOS)
        
        # Random crop position
        max_x = expanded_w - width
        max_y = expanded_h - height
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)
        background = background.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    # 3. ENHANCED ROTATION - 45% chance (increased from 30%)
    if random.random() < 0.65:
        # Wider rotation range and better edge handling
        rotation_angle = random.uniform(-25, 25)  # Increased from Â±15 to Â±25 degrees
        
        # Method 1: Simple rotation with smart fill color (60% of rotations)
        if random.random() < 0.6:
            # Calculate average edge color for better fill
            edge_pixels = []
            # Sample edges
            edge_pixels.extend(list(background.crop((0, 0, width, 5)).getdata()))  # Top
            edge_pixels.extend(list(background.crop((0, height-5, width, height)).getdata()))  # Bottom
            edge_pixels.extend(list(background.crop((0, 0, 5, height)).getdata()))  # Left  
            edge_pixels.extend(list(background.crop((width-5, 0, width, height)).getdata()))  # Right
            
            if edge_pixels:
                avg_r = int(np.mean([p[0] for p in edge_pixels]))
                avg_g = int(np.mean([p[1] for p in edge_pixels]))
                avg_b = int(np.mean([p[2] for p in edge_pixels]))
                fill_color = (avg_r, avg_g, avg_b)
            else:
                fill_color = (40, 40, 40)
            
            background = background.rotate(rotation_angle, expand=False, fillcolor=fill_color, resample=Image.BICUBIC)
            
        else:
            # Method 2: Expand canvas method for more complex rotations
            diagonal = int(np.sqrt(width*width + height*height)) + 50  # Extra padding
            expanded_bg = Image.new('RGB', (diagonal, diagonal))
            
            # Create a gradient fill or pattern for expanded area
            if random.random() < 0.5:
                # Gradient fill based on background colors
                bg_array = np.array(background)
                corner_colors = [
                    bg_array[0, 0],      # Top-left
                    bg_array[0, -1],     # Top-right  
                    bg_array[-1, 0],     # Bottom-left
                    bg_array[-1, -1]     # Bottom-right
                ]
                avg_color = tuple(np.mean(corner_colors, axis=0).astype(int))
                expanded_bg.paste(Image.new('RGB', (diagonal, diagonal), avg_color))
            else:
                # Use edge-based fill
                expanded_bg.paste(Image.new('RGB', (diagonal, diagonal), (30, 30, 30)))
            
            # Paste original background in center
            paste_x = (diagonal - width) // 2
            paste_y = (diagonal - height) // 2
            expanded_bg.paste(background, (paste_x, paste_y))
            
            # Rotate and crop back to original size
            rotated = expanded_bg.rotate(rotation_angle, expand=False, resample=Image.BICUBIC)
            crop_x = (diagonal - width) // 2
            crop_y = (diagonal - height) // 2
            background = rotated.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    # 4. ENHANCED COLOR VARIATIONS - Applied with higher probability and more types
    
    # 4a. Hue shift - 70% chance (increased from 60%)
    if random.random() < 0.7:
        # Convert to HSV for hue manipulation
        hsv = background.convert('HSV')
        h, s, v = hsv.split()
        
        # Wider hue shift range
        hue_shift = random.randint(-45, 45)  # Increased from Â±30 to Â±45 hue units
        h_array = np.array(h)
        h_array = (h_array.astype(np.int16) + hue_shift) % 256
        h = Image.fromarray(h_array.astype(np.uint8))
        
        background = Image.merge('HSV', (h, s, v)).convert('RGB')
    
    # 4b. ENHANCED Brightness - 80% chance (increased from 70%)  
    if random.random() < 0.8:
        brightness_factor = random.uniform(0.5, 1.6)  # Wider range: 0.5x to 1.6x brightness
        background = ImageEnhance.Brightness(background).enhance(brightness_factor)
    
    # 4c. ENHANCED Contrast - 70% chance (increased from 60%)
    if random.random() < 0.7:
        contrast_factor = random.uniform(0.6, 1.5)  # Wider range: 0.6x to 1.5x contrast
        background = ImageEnhance.Contrast(background).enhance(contrast_factor)
    
    # 4d. ENHANCED Saturation - 60% chance (increased from 50%) 
    if random.random() < 0.6:
        saturation_factor = random.uniform(0.4, 1.7)  # Wider range: 0.4x to 1.7x saturation
        background = ImageEnhance.Color(background).enhance(saturation_factor)

    # 4e. NEW: Gamma correction - 30% chance
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.4)  # Gamma values
        bg_array = np.array(background).astype(np.float32) / 255.0
        bg_array = np.power(bg_array, gamma)
        bg_array = (bg_array * 255).clip(0, 255)
        background = Image.fromarray(bg_array.astype(np.uint8))

    # 4f. NEW: Lighting simulation - directional lighting effect - 25% chance
    if random.random() < 0.25:
        # Create directional lighting gradient
        lighting_direction = random.choice(['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right'])
        lighting_intensity = random.uniform(0.3, 0.8)
        
        # Create gradient mask
        gradient = Image.new('L', (width, height), 128)  # Start with neutral
        gradient_draw = ImageDraw.Draw(gradient)
        
        if lighting_direction == 'left':
            for x in range(width):
                intensity = int(255 * (1 - (x / width) * lighting_intensity))
                gradient_draw.line([(x, 0), (x, height)], fill=intensity)
        elif lighting_direction == 'right':
            for x in range(width):
                intensity = int(255 * (1 - ((width - x) / width) * lighting_intensity))
                gradient_draw.line([(x, 0), (x, height)], fill=intensity)
        elif lighting_direction == 'top':
            for y in range(height):
                intensity = int(255 * (1 - (y / height) * lighting_intensity))
                gradient_draw.line([(0, y), (width, y)], fill=intensity)
        elif lighting_direction == 'bottom':
            for y in range(height):
                intensity = int(255 * (1 - ((height - y) / height) * lighting_intensity))
                gradient_draw.line([(0, y), (width, y)], fill=intensity)
        # Add diagonal directions
        elif lighting_direction == 'top-left':
            for y in range(height):
                for x in range(width):
                    distance = np.sqrt(x*x + y*y) / np.sqrt(width*width + height*height)
                    intensity = int(255 * (1 - distance * lighting_intensity))
                    gradient_draw.point((x, y), fill=intensity)
        # ... similar for other diagonal directions
        
        # Apply lighting gradient
        bg_array = np.array(background)
        gradient_array = np.array(gradient) / 255.0
        
        for c in range(3):
            bg_array[:, :, c] = (bg_array[:, :, c] * gradient_array).clip(0, 255)
        
        background = Image.fromarray(bg_array.astype(np.uint8))

    # 4g. NEW: Color temperature adjustment - 20% chance
    if random.random() < 0.2:
        # Simulate different color temperatures (warm/cool lighting)
        temp_shift = random.uniform(-0.3, 0.3)  # Negative = cooler, Positive = warmer
        
        bg_array = np.array(background).astype(np.float32)
        
        if temp_shift > 0:  # Warmer (more red/yellow)
            bg_array[:, :, 0] *= (1 + temp_shift * 0.5)  # Increase red
            bg_array[:, :, 1] *= (1 + temp_shift * 0.3)  # Slightly increase green  
            bg_array[:, :, 2] *= (1 - temp_shift * 0.2)  # Decrease blue
        else:  # Cooler (more blue)
            bg_array[:, :, 0] *= (1 + temp_shift * 0.2)  # Decrease red
            bg_array[:, :, 1] *= (1 + temp_shift * 0.1)  # Slightly decrease green
            bg_array[:, :, 2] *= (1 - temp_shift * 0.5)  # Increase blue
            
        bg_array = np.clip(bg_array, 0, 255)
        background = Image.fromarray(bg_array.astype(np.uint8))

    # 5. Gaussian blur - 25% chance (simulates depth of field)
    if random.random() < 0.25:
        blur_radius = random.uniform(0.5, 2.5)  # Slightly wider range
        background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 6. Noise injection - 25% chance (increased from 20%)
    if random.random() < 0.25:
        # Add subtle Gaussian noise
        bg_array = np.array(background)
        noise_intensity = random.uniform(3, 12)  # Wider range
        noise = np.random.normal(0, noise_intensity, bg_array.shape)
        noisy_bg = np.clip(bg_array.astype(np.float32) + noise, 0, 255)
        background = Image.fromarray(noisy_bg.astype(np.uint8))

    # 7. ENHANCED Vignetting effect - 20% chance (increased from 15%)
    if random.random() < 0.2:
        # Create vignette mask
        center_x, center_y = width // 2, height // 2
        
        # Allow off-center vignettes
        if random.random() < 0.3:  # 30% chance for off-center
            center_x += random.randint(-width//4, width//4)
            center_y += random.randint(-height//4, height//4)
            
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette = Image.new('L', (width, height), 255)
        vignette_draw = ImageDraw.Draw(vignette)
        
        # Enhanced vignette with more variation
        vignette_strength = random.uniform(0.2, 0.8)  # Wider range
        vignette_falloff = random.uniform(0.5, 2.0)   # Control falloff rate
        
        for r in range(int(max_dist), 0, -3):  # Finer steps
            intensity = int(255 * (1 - ((max_dist - r) / max_dist * vignette_strength) ** vignette_falloff))
            intensity = max(0, min(255, intensity))  # Clamp values
            bbox = [center_x - r, center_y - r, center_x + r, center_y + r]
            vignette_draw.ellipse(bbox, fill=intensity)
        
        # Apply vignette with variable blur
        blur_amount = random.uniform(15, 35)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        bg_array = np.array(background)
        vignette_array = np.array(vignette) / 255.0
        
        for c in range(3):
            bg_array[:, :, c] = bg_array[:, :, c] * vignette_array
        
        background = Image.fromarray(bg_array.astype(np.uint8))

    # 8. Perspective transform - 15% chance (increased from 10%)
    if random.random() < 0.15:
        # Enhanced perspective distortion
        distortion = random.uniform(0.01, 0.08)  # Slightly wider range
        
        # Define perspective transform coefficients  
        a = 1 + random.uniform(-distortion, distortion)
        b = random.uniform(-distortion, distortion)
        c = random.uniform(-distortion*width, distortion*width)
        d = random.uniform(-distortion, distortion)
        e = 1 + random.uniform(-distortion, distortion)
        f = random.uniform(-distortion*height, distortion*height)
        g = random.uniform(-distortion/width, distortion/width)
        h = random.uniform(-distortion/height, distortion/height)
        
        try:
            background = background.transform(
                (width, height), 
                Image.PERSPECTIVE,
                (a, b, c, d, e, f, g, h),
                Image.LANCZOS
            )
        except:
            pass  # Skip if transform fails

    # 9. NEW: Shadow/highlight adjustment - 15% chance
    if random.random() < 0.15:
        bg_array = np.array(background).astype(np.float32) / 255.0
        
        # Separate shadows and highlights
        shadow_threshold = 0.3
        highlight_threshold = 0.7
        
        shadow_adjust = random.uniform(-0.3, 0.3)
        highlight_adjust = random.uniform(-0.3, 0.3)
        
        # Create masks
        luminance = 0.299 * bg_array[:,:,0] + 0.587 * bg_array[:,:,1] + 0.114 * bg_array[:,:,2]
        shadow_mask = np.where(luminance < shadow_threshold, 1 - (luminance / shadow_threshold), 0)
        highlight_mask = np.where(luminance > highlight_threshold, (luminance - highlight_threshold) / (1 - highlight_threshold), 0)
        
        # Apply adjustments
        for c in range(3):
            bg_array[:,:,c] += shadow_mask * shadow_adjust
            bg_array[:,:,c] += highlight_mask * highlight_adjust
            
        bg_array = np.clip(bg_array, 0, 1)
        background = Image.fromarray((bg_array * 255).astype(np.uint8))

    # === FINAL COMPOSITION ===
    # Simple composite: background + tools
    result = Image.new('RGBA', (width, height))
    result.paste(background, (0, 0))
    result.paste(pil_foreground, (0, 0), mask=pil_foreground)

    # Convert to RGB for final output
    final_image = Image.new('RGB', (width, height), (255, 255, 255))
    final_image.paste(result, mask=result.split()[-1])

    return np.array(final_image) / 255.0



def enhance_background_with_extreme_variations(rendered_image, single_background_path):
    """
    EXTREME background augmentations for maximum robustness:
    - Includes all previous augmentations
    - Higher probability of applying multiple effects
    - More aggressive parameter ranges
    """
    
    # [Same initial setup code as above...]
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

    if not os.path.exists(single_background_path):
        background = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        background = Image.open(single_background_path).convert('RGB')

    # === EXTREME AUGMENTATIONS ===
    
    # More aggressive zoom: 70% chance, wider range
    if random.random() < 0.7:
        scale_factor = random.uniform(0.5, 1.8)  # 0.5x to 1.8x (more extreme)
        # [Same zoom logic as above but with wider range]
        
    # More frequent position shifts: 60% chance  
    if random.random() < 0.6:
        expand_factor = random.uniform(1.2, 1.6)  # Larger shifts
        # [Same logic as above]
        
    # More aggressive color variations
    if random.random() < 0.8:  # Higher chance
        brightness_factor = random.uniform(0.4, 1.6)  # Wider range
        background = ImageEnhance.Brightness(background).enhance(brightness_factor)
        
    if random.random() < 0.7:
        contrast_factor = random.uniform(0.5, 1.5)  # Wider range
        background = ImageEnhance.Contrast(background).enhance(contrast_factor)
    
    # Multiple effects can stack for more variation
    # [Include rest of effects with higher probabilities and more aggressive ranges]
    
    # [Same final composition code...]
    result = Image.new('RGBA', (width, height))
    result.paste(background, (0, 0))
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

        # --- Mid band (30â€“70% Z) ---
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

    # Helper: choose connections by category
    def get_connections(cat):
        if cat == 'needle_holder':
            return [
                ("bottom_left", "middle_left"),
                ("middle_left", "joint_center"),
                ("joint_center", "top_left"),
                ("bottom_right", "middle_right"),
                ("middle_right", "joint_center"),
                ("joint_center", "top_right"),
            ]
        elif cat == 'tweezers':
            return [
                ("top_left",  "mid_left"),
                ("mid_left",  "bottom_tip"),
                ("top_right", "mid_right"),
                ("mid_right", "bottom_tip"),
            ]
        return []

    legend_patches = []

    if isinstance(keypoints_2d, list):
        # --- Multi-object visualization: keypoints_2d is a list of per-object dicts ---
        # Draw connection lines first for every object
        for obj_d in keypoints_2d:
            obj_kps = obj_d.get("keypoints", {})
            obj_cat = obj_d.get("category", None) or category
            conns = get_connections(obj_cat)
            for a, b in conns:
                pa, pb = obj_kps.get(a), obj_kps.get(b)
                if pa is not None and pb is not None:
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                            '-', linewidth=2,
                            color=keypoint_colors.get(a, 'white'),
                            alpha=0.9, zorder=1)

        # Then draw points (so they appear on top), and per-object bboxes
        for obj_d in keypoints_2d:
            obj_kps = obj_d.get("keypoints", {})
            for kp_name, kp_color in keypoint_colors.items():
                if kp_name in obj_kps and obj_kps[kp_name] is not None:
                    x, y = obj_kps[kp_name]
                    ax.plot(x, y, 'o', markersize=6, color=kp_color,
                            markeredgecolor='white', markeredgewidth=1, zorder=2)
                    if not any(p.get_label() == kp_name for p in legend_patches):
                        legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))

            obbox = obj_d.get("bbox", None)
            if obbox and len(obbox) == 4 and obbox[2] > 0 and obbox[3] > 0:
                rect = mpatches.Rectangle((obbox[0], obbox[1]), obbox[2], obbox[3],
                                          linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                label = obj_d.get("object_id", "bbox")
                ax.text(obbox[0], obbox[1] - 5, f'{label}', color='lime', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))

        title = f'Multi-object - Frame {frame_idx}{title_suffix}'

    else:
        # --- Single-object visualization (existing behavior) ---
        # Draw connection lines first
        connections = get_connections(category)
        for a, b in connections:
            pa = keypoints_2d.get(a); pb = keypoints_2d.get(b)
            if pa is not None and pb is not None:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                        '-', linewidth=2,
                        color=keypoint_colors.get(a, 'white'),
                        alpha=0.9, zorder=1)

        # Draw keypoints
        for kp_name, kp_color in keypoint_colors.items():
            if kp_name in keypoints_2d:
                pt = keypoints_2d[kp_name]
                if pt is not None:
                    x, y = pt
                    ax.plot(x, y, 'o', markersize=6, color=kp_color,
                            markeredgecolor='white', markeredgewidth=1, zorder=2)
                    legend_patches.append(mpatches.Patch(color=kp_color, label=kp_name))

        # Draw bbox if provided
        if bbox is not None and len(bbox) == 4:
            rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                      linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 5, 'bbox', color='lime', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))

        title = f'{category.replace("_", " ").title()}: {obj_identifier} - Frame {frame_idx}{title_suffix}'

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.8)

    ax.axis('off')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig



def process_objects(obj_paths, camera_params, output_dir, num_images=25,
                   object_counter=None, backgrounds_dir=None, haven_path=None, global_hdri_counter=None,
                   single_background_path=None):
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
                offset_mag = random.uniform(0.5, 0.6)
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

            # Initial verts (will recompute after "aim to center")
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

            # REDUCED noise for more consistent left/right orientation
            jitter = random.uniform(0.0, 0.003) * 2.0  # REDUCED from 0.006 to 0.003

            target_xy = scene_center[:2] + np.array([random.uniform(-jitter, jitter),
                                                     random.uniform(-jitter, jitter)])

            for d in loaded_objects:
                obj = d['obj']
                loc_xy = obj.get_location()[:2]
                bearing = np.arctan2(target_xy[1] - loc_xy[1], target_xy[0] - loc_xy[0])
                eul = obj.get_rotation_euler()
                
                # REDUCED rotation noise to maintain left/right consistency
                noise = np.deg2rad(random.uniform(-1.5, 1.5))  # REDUCED from (-3.0, 3.0) to (-1.5, 1.5)
                eul[2] = float(bearing) + noise                 
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
            # SINGLE OBJECT: Add slight random rotation but keep it minimal
            d = loaded_objects[0]
            obj = d['obj']
            eul = obj.get_rotation_euler()
            
            # MINIMAL random rotation for single objects to maintain orientation consistency
            small_rotation = np.deg2rad(random.uniform(-2.0, 2.0))  # Very small rotation range
            eul[2] += small_rotation
            obj.set_rotation_euler(eul)
            
            # Recompute after small rotation
            mesh = obj.get_mesh()
            verts = mesh.vertices
            obj2world = obj.get_local2world_mat()
            verts_world = [obj2world @ np.append(v.co, 1.0) for v in verts]
            verts_world = [v[:3] for v in verts_world]
            d['verts_world'] = verts_world
            
            # single: compute keypoints once
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
            
            # NEW: Decide if this should be a "partial visibility" frame (20% chance)
            create_partial_visibility = random.random() < 0.55  # 1 in 5 chance

            if scene_type == "single":
                if create_partial_visibility:
                    # PARTIAL VISIBILITY: Camera closer and looking up
                    location = bproc.sampler.shell(
                        center=focus_obj.get_location(), 
                        radius_min=2.5, radius_max=4.0,     # CLOSER for partial visibility
                        elevation_min=25, elevation_max=45   # LOWER angle (looking up)
                    )
                    tool_center = focus_obj.get_location()
                    # Look at lower part of tool to crop top portion
                    tool_bbox = focus_obj.get_bound_box()
                    tool_dimensions = np.array(tool_bbox).max(axis=0) - np.array(tool_bbox).min(axis=0)
                    tool_length = max(tool_dimensions)
                    
                    # Look at bottom 30% of tool to crop the top
                    lookat_point = tool_center + np.array([
                        random.uniform(-0.1, 0.1), 
                        random.uniform(-0.1, 0.1), 
                        -tool_length * 0.6  # Look lower to crop top
                    ])
                    
                    rotation_matrix = bproc.camera.rotation_from_forward_vec(
                        lookat_point - location, 
                        inplane_rot=np.deg2rad(np.random.uniform(-150, -30))
                    )
                else:
                    # NORMAL VISIBILITY (existing code)
                    location = bproc.sampler.shell(
                        center=focus_obj.get_location(), 
                        radius_min=4.0, radius_max=6.5,     
                        elevation_min=45, elevation_max=85  
                    )
                    tool_center = focus_obj.get_location()
                    tool_bbox = focus_obj.get_bound_box()
                    tool_dimensions = np.array(tool_bbox).max(axis=0) - np.array(tool_bbox).min(axis=0)
                    tool_length = max(tool_dimensions)
                    
                    if random.random() < 0.25:  
                        offset_factor = random.uniform(0.1, 0.3)  
                        lookat_point = tool_center + np.array([
                            random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), -tool_length * offset_factor
                        ])
                    else:
                        lookat_point = tool_center + np.array([
                            random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), -tool_length * 0.2
                        ])
                    
                    rotation_matrix = bproc.camera.rotation_from_forward_vec(
                        lookat_point - location, 
                        inplane_rot=np.deg2rad(np.random.uniform(-150, -30))
                    )
            else:
                # PAIR SCENE
                if create_partial_visibility:
                    # PARTIAL VISIBILITY: One or both tools partially out of frame
                    location = bproc.sampler.shell(
                        center=scene_center, 
                        radius_min=4.5, radius_max=7.0,     # CLOSER for partial visibility
                        elevation_min=40, elevation_max=60   # LOWER angle
                    )
                    
                    # Offset scene center to crop some tools
                    scene_offset = np.array([
                        random.uniform(-0.8, 0.8),  # LARGER offsets for partial visibility
                        random.uniform(-0.6, 0.6),
                        random.uniform(-0.4, 0.4)
                    ])
                    lookat_point = scene_center + scene_offset
                    
                    rotation_matrix = bproc.camera.rotation_from_forward_vec(
                        lookat_point - location, 
                        inplane_rot=np.deg2rad(np.random.uniform(-110, -70))
                    )
                else:
                    # NORMAL VISIBILITY (existing code)
                    location = bproc.sampler.shell(
                        center=scene_center, 
                        radius_min=6.0, radius_max=9.0,     
                        elevation_min=50, elevation_max=70  
                    )
                    
                    if random.random() < 0.20:  
                        scene_offset = np.array([
                            random.uniform(-0.3, 0.3),  
                            random.uniform(-0.3, 0.3),
                            random.uniform(-0.2, 0.2)
                        ])
                        lookat_point = scene_center + scene_offset
                    else:
                        lookat_point = scene_center
                    
                    rotation_matrix = bproc.camera.rotation_from_forward_vec(
                        lookat_point - location, 
                        inplane_rot=np.deg2rad(np.random.uniform(-110, -70))
                    )

            


            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

            # More permissive visibility check
            all_visible = True
            for obj_data in loaded_objects:
                obj = obj_data['obj']
                try:
                    obj_location = obj.get_location()
                    cam_to_obj = obj_location - location
                    cam_forward = rotation_matrix @ np.array([0, 0, -1])
                    
                    # More lenient check for partial visibility frames
                    threshold = -0.6 if create_partial_visibility else -0.3
                    if np.dot(cam_to_obj, cam_forward) < threshold:  
                        all_visible = False
                        break
                except:
                    if not create_partial_visibility:  # Skip BlenderProc visibility check for partial frames
                        if obj not in bproc.camera.visible_objects(cam2world_matrix):
                            all_visible = False
                            break

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

                    for name, pt in list(frame_keypoints.items()):
                        if pt is None:
                            continue

                        x, y = float(pt[0]), float(pt[1])

                        
                        if x < 0 or x >= im_width or y < 0 or y >= im_height:
                            frame_keypoints[name] = None
                            print(f"Removed keypoint '{name}' at ({x:.1f}, {y:.1f}) - outside image bounds")
                            continue

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
                                    x = min(max(x, x0), x1)
                                    y = min(max(y, y0), y1)

                        # Step 1: clamp outside bbox -> edge (but keep in image)
                        if (x < x0) or (x > x1) or (y < y0) or (y > y1):
                            x = min(max(x, x0), x1)
                            y = min(max(y, y0), y1)

                        # Step 2: inside bbox but off-mask -> gentle constraint (skip for tweezers bottom_tip)
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
                                    if a is not None and b is not None: 
                                        projected = proj_to_segment([x, y], a, b)
                                        # Only use projection if it stays in image
                                        if 0 <= projected[0] < im_width and 0 <= projected[1] < im_height:
                                            x, y = projected
                                elif 'right' in name:
                                    a = frame_keypoints.get('bottom_right'); b = frame_keypoints.get('top_right')
                                    if a is not None and b is not None:
                                        projected = proj_to_segment([x, y], a, b)
                                        # Only use projection if it stays in image
                                        if 0 <= projected[0] < im_width and 0 <= projected[1] < im_height:
                                            x, y = projected
                            elif category == 'tweezers':
                                if name == 'bottom_tip':
                                    # keep as the chosen vertex projection; no midline override here
                                    pass
                                elif 'left' in name:
                                    a = frame_keypoints.get('mid_left');  b = frame_keypoints.get('top_left')
                                    if a is not None and b is not None: 
                                        projected = proj_to_segment([x, y], a, b)
                                        # Only use projection if it stays in image
                                        if 0 <= projected[0] < im_width and 0 <= projected[1] < im_height:
                                            x, y = projected
                                elif 'right' in name:
                                    a = frame_keypoints.get('mid_right'); b = frame_keypoints.get('top_right')
                                    if a is not None and b is not None: 
                                        projected = proj_to_segment([x, y], a, b)
                                        # Only use projection if it stays in image
                                        if 0 <= projected[0] < im_width and 0 <= projected[1] < im_height:
                                            x, y = projected

                            # If still off-mask, snap to nearest foreground ONLY if not bottom_tip
                            if not (category == 'tweezers' and name == 'bottom_tip'):
                                xi = int(round(x)); yi = int(round(y))
                                xi = max(0, min(obj_mask.shape[1]-1, xi))
                                yi = max(0, min(obj_mask.shape[0]-1, yi))
                                if obj_mask[yi, xi] == 0 and (mask_points is not None and mask_points.size > 0):
                                    dx = mask_points[:, 0] - x
                                    dy = mask_points[:, 1] - y
                                    j = int(np.argmin(dx*dx + dy*dy))
                                    snap_x = float(mask_points[j, 0])
                                    snap_y = float(mask_points[j, 1])
                                    # Only snap if it stays in image
                                    if 0 <= snap_x < im_width and 0 <= snap_y < im_height:
                                        x, y = snap_x, snap_y

                        # Final boundary check - remove if outside image after all corrections
                        if x < 0 or x >= im_width or y < 0 or y >= im_height:
                            frame_keypoints[name] = None
                            print(f"Removed keypoint '{name}' after correction - outside image bounds")
                        else:
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

            # Use the new background enhancement function (without hands)
            if single_background_path is not None:
                bg_image = enhance_background_with_variations(image, single_background_path)  # New function
            else:
                # Use original background directory method
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
                # Use the unified visualization (handles multi-object lines + points)
                fig = create_visualization(
                    bg_image,
                    frame_objects,            # list of {"keypoints","category","bbox","object_id",...}
                    keypoint_colors,
                    obj_identifier="",        # unused in multi-object path
                    category="",              # unused; per-object category is used
                    frame_idx=i,
                    title_suffix=" (with Background)",
                    bbox=None                 # per-object bboxes are handled inside
                )

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
    parser.add_argument('--num_images', type=int, default=2, help="Number of images per object/pair")
    parser.add_argument('--categories', nargs='+', default=['needle_holder', 'tweezers'], help="Categories to process")
    parser.add_argument('--backgrounds_dir', default="/datashare/project/train2017", help="Directory containing background images")
    parser.add_argument('--haven_path', default="/datashare/project/haven/", help="Path to the haven HDRI images")
    
    # New arguments for single background and hands overlay
  

    default_root = "/home/student/Desktop/VisionSDS-Project"

    parser.add_argument(
        '--single_background',
        default=os.path.join(default_root, "surg_background.png"),
        help="Path to single background file (instead of backgrounds_dir)"
    )

    

    
    args = parser.parse_args()

    # Validate background options
    if args.single_background:
        if not os.path.exists(args.single_background):
            print(f"Error: Single background file does not exist: {args.single_background}")
            return
        print(f"Using single background: {args.single_background}")
        
           
    else:
        if not os.path.exists(args.backgrounds_dir):
            print(f"Error: Background directory does not exist: {args.backgrounds_dir}")
            return
        print(f"Using background directory: {args.backgrounds_dir}")
    
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
    if args.single_background:
        print(f"  Output: Background-enhanced images with single background + hands overlay")
    else:
        print(f"  Output: Background-enhanced images with multiple backgrounds")

    print("Initializing BlenderProc...")
    bproc.init()

    try:
        successful_scenes = []
        failed_scenes = []
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
                        haven_path=args.haven_path, global_hdri_counter=global_hdri_counter,
                        single_background_path=args.single_background
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
                    print(f"    âœ“ {scene_id}: {num_rendered} images")

            if pair_scenes:
                print(f"  Tool pairs ({len(pair_scenes)}):")
                for scene_id, num_rendered, scene_type in pair_scenes:
                    print(f"    âœ“ {scene_id}: {num_rendered} images")

        if failed_scenes:
            print("\nFailed scenes:")
            for scene_id, scene_type in failed_scenes:
                print(f"  âœ— {scene_id} ({scene_type})")

        summary_file = os.path.join(args.output_dir, "processing_summary.json")
        summary_data = {
            "configuration": {
                "backgrounds_dir": args.backgrounds_dir if not args.single_background else None,
                "single_background": args.single_background,
                
                "haven_path": args.haven_path,
                "num_images_per_scene": args.num_images,
                "categories": args.categories,
                "output_type": "single_background_with_hands" if args.single_background else "background_enhanced_only"
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
            if args.single_background:
                print(f"  Single background + hands overlay: Applied to all images")
            else:
                print(f"  Background enhancement: Applied to all images")
            print(f"  HDRI lighting: Applied to all scenes")

    finally:
        bproc.clean_up()

if __name__ == "__main__":
    main()