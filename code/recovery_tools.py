"""
Recovery tools for image restoration and enhancement.

This module provides a collection of traditional image processing tools
that can be used to recover images from various corruptions (blur, noise,
color distortions, etc.) applied during robustness evaluation.

Each tool is designed to counter specific types of image degradation
while being lightweight (PIL/scipy-based, no deep learning models).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


@dataclass
class RecoveryTool:
    """Defines a recovery tool with metadata for VLM selection.

    Attributes:
        name: Unique identifier for the tool
        display_name: Human-readable name for display
        description: Description for VLM to understand when to use this tool
        category: Tool category (blur, noise, color, geometric, weather, compression, resolution)
        function: The actual recovery function (Image, params) -> Image
        parameters: Parameter specifications with type, default, min, max, description
        counters: List of augmentation names this tool can counter
    """
    name: str
    display_name: str
    description: str
    category: str
    function: Callable[[Image.Image, Dict[str, Any]], Image.Image]
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counters: List[str] = field(default_factory=list)

    def __call__(self, image: Image.Image, params: Dict[str, Any] = None) -> Image.Image:
        """Apply the tool to an image with optional parameters."""
        params = params or {}
        # Merge with defaults
        final_params = {}
        for param_name, param_info in self.parameters.items():
            final_params[param_name] = params.get(param_name, param_info.get('default'))
        return self.function(image, final_params)

    def get_schema_for_vlm(self) -> Dict[str, Any]:
        """Return JSON schema for structured output parsing."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                name: {
                    "type": info["type"],
                    "description": info.get("description", ""),
                    "default": info.get("default"),
                }
                for name, info in self.parameters.items()
            }
        }


# =============================================================================
# BLUR RECOVERY TOOLS
# =============================================================================

def sharpen_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Sharpen blurry images using PIL's sharpness enhancer."""
    factor = params.get('factor', 2.0)
    return ImageEnhance.Sharpness(image.convert('RGB')).enhance(factor)


def unsharp_mask_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Apply unsharp mask for edge enhancement."""
    radius = params.get('radius', 2)
    percent = params.get('percent', 150)
    threshold = params.get('threshold', 3)
    return image.convert('RGB').filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )


def high_pass_sharpen_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """High-pass filter based sharpening for severe blur."""
    strength = params.get('strength', 1.5)

    arr = np.array(image.convert('RGB'), dtype=np.float32) / 255.0

    # Apply Gaussian blur to get low frequencies
    if len(arr.shape) == 3:
        blurred = np.stack([gaussian_filter(arr[:,:,c], sigma=1.0) for c in range(3)], axis=-1)
    else:
        blurred = gaussian_filter(arr, sigma=1.0)

    # High-pass = original - blurred
    high_pass = arr - blurred

    # Add high frequencies back with strength
    sharpened = arr + strength * high_pass
    sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(sharpened)


BLUR_RECOVERY_TOOLS = [
    RecoveryTool(
        name="sharpen",
        display_name="Sharpen",
        description="Increase image sharpness. Use when image appears blurry, soft, or out of focus. Good for mild to moderate blur.",
        category="blur",
        function=sharpen_tool,
        parameters={
            "factor": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 5.0,
                "description": "Sharpening intensity (1.0=none, 2.0=moderate, 5.0=strong)"
            }
        },
        counters=["gaussian_blur", "defocus_blur", "glass_blur"]
    ),
    RecoveryTool(
        name="unsharp_mask",
        display_name="Unsharp Mask",
        description="Edge enhancement filter. Use for images where edges and details need to be more defined. Better for moderate to severe blur than simple sharpen.",
        category="blur",
        function=unsharp_mask_tool,
        parameters={
            "radius": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 10,
                "description": "Blur radius for the mask"
            },
            "percent": {
                "type": "int",
                "default": 150,
                "min": 50,
                "max": 300,
                "description": "Sharpening strength percentage"
            },
            "threshold": {
                "type": "int",
                "default": 3,
                "min": 0,
                "max": 10,
                "description": "Minimum brightness change to sharpen"
            }
        },
        counters=["gaussian_blur", "defocus_blur", "motion_blur", "zoom_blur"]
    ),
    RecoveryTool(
        name="high_pass_sharpen",
        display_name="High-Pass Sharpen",
        description="Aggressive sharpening using high-pass filter. Use for severely blurred images where other sharpening methods are insufficient.",
        category="blur",
        function=high_pass_sharpen_tool,
        parameters={
            "strength": {
                "type": "float",
                "default": 1.5,
                "min": 0.5,
                "max": 3.0,
                "description": "High-pass filter strength"
            }
        },
        counters=["gaussian_blur", "motion_blur", "defocus_blur", "zoom_blur", "glass_blur"]
    ),
]


# =============================================================================
# NOISE RECOVERY TOOLS
# =============================================================================

def median_filter_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Median filter for salt-and-pepper noise removal."""
    size = params.get('size', 3)
    arr = np.array(image.convert('RGB'))
    if len(arr.shape) == 3:
        filtered = np.stack([median_filter(arr[:,:,c], size=size) for c in range(3)], axis=-1)
    else:
        filtered = median_filter(arr, size=size)
    return Image.fromarray(filtered.astype(np.uint8))


def gaussian_smooth_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Gaussian smoothing for Gaussian noise reduction."""
    sigma = params.get('sigma', 1.0)
    arr = np.array(image.convert('RGB'), dtype=np.float32)
    if len(arr.shape) == 3:
        smoothed = np.stack([gaussian_filter(arr[:,:,c], sigma=sigma) for c in range(3)], axis=-1)
    else:
        smoothed = gaussian_filter(arr, sigma=sigma)
    return Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8))


def bilateral_filter_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Edge-preserving bilateral filter using iterative approximation."""
    sigma_spatial = params.get('sigma_spatial', 5.0)
    sigma_range = params.get('sigma_range', 0.1)
    iterations = params.get('iterations', 2)

    arr = np.array(image.convert('RGB'), dtype=np.float32) / 255.0

    for _ in range(iterations):
        # Spatial Gaussian blur
        if len(arr.shape) == 3:
            blurred = np.stack([gaussian_filter(arr[:,:,c], sigma=sigma_spatial / 3) for c in range(3)], axis=-1)
        else:
            blurred = gaussian_filter(arr, sigma=sigma_spatial / 3)

        # Edge-preserving blend based on intensity difference
        diff = np.abs(arr - blurred)
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2, keepdims=True)
        weight = np.exp(-diff / sigma_range)
        arr = arr * weight + blurred * (1 - weight)

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


NOISE_RECOVERY_TOOLS = [
    RecoveryTool(
        name="median_filter",
        display_name="Median Filter",
        description="Removes salt-and-pepper noise (random black and white speckles). Preserves edges better than Gaussian smoothing. Good for impulsive noise.",
        category="noise",
        function=median_filter_tool,
        parameters={
            "size": {
                "type": "int",
                "default": 3,
                "min": 3,
                "max": 7,
                "description": "Filter kernel size (odd number, larger = more smoothing)"
            }
        },
        counters=["salt_pepper", "speckle_noise"]
    ),
    RecoveryTool(
        name="gaussian_smooth",
        display_name="Gaussian Smoothing",
        description="Reduces Gaussian noise (grainy appearance). Good for general noise but may slightly blur edges. Use with mild settings to preserve detail.",
        category="noise",
        function=gaussian_smooth_tool,
        parameters={
            "sigma": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 3.0,
                "description": "Smoothing strength (higher = more smoothing but more blur)"
            }
        },
        counters=["gaussian_noise", "shot_noise"]
    ),
    RecoveryTool(
        name="bilateral_filter",
        display_name="Bilateral Filter",
        description="Edge-preserving noise reduction. Smooths noise while keeping edges sharp. Best for noisy images where preserving edge detail is important.",
        category="noise",
        function=bilateral_filter_tool,
        parameters={
            "sigma_spatial": {
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 10.0,
                "description": "Spatial smoothing range"
            },
            "sigma_range": {
                "type": "float",
                "default": 0.1,
                "min": 0.05,
                "max": 0.3,
                "description": "Intensity range for edge preservation (lower = sharper edges)"
            }
        },
        counters=["gaussian_noise", "shot_noise", "speckle_noise"]
    ),
]


# =============================================================================
# COLOR RECOVERY TOOLS
# =============================================================================

def auto_contrast_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Automatic contrast stretching."""
    cutoff = params.get('cutoff', 0)
    return ImageOps.autocontrast(image.convert('RGB'), cutoff=cutoff)


def gamma_correction_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Gamma correction for brightness adjustment."""
    gamma = params.get('gamma', 1.0)
    inv_gamma = 1.0 / gamma
    lut = [int((i / 255.0) ** inv_gamma * 255) for i in range(256)]
    img = image.convert('RGB')
    return img.point(lut * 3)


def brightness_adjust_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Adjust image brightness."""
    factor = params.get('factor', 1.0)
    return ImageEnhance.Brightness(image.convert('RGB')).enhance(factor)


def contrast_adjust_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Adjust image contrast."""
    factor = params.get('factor', 1.0)
    return ImageEnhance.Contrast(image.convert('RGB')).enhance(factor)


def saturation_adjust_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Adjust color saturation."""
    factor = params.get('factor', 1.0)
    return ImageEnhance.Color(image.convert('RGB')).enhance(factor)


def histogram_equalize_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Histogram equalization for contrast enhancement."""
    return ImageOps.equalize(image.convert('RGB'))


def white_balance_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """White balance correction using gray world assumption."""
    arr = np.array(image.convert('RGB'), dtype=np.float32)

    avg_r = arr[:,:,0].mean()
    avg_g = arr[:,:,1].mean()
    avg_b = arr[:,:,2].mean()
    avg_gray = (avg_r + avg_g + avg_b) / 3

    # Avoid division by zero
    arr[:,:,0] *= avg_gray / (avg_r + 1e-6)
    arr[:,:,1] *= avg_gray / (avg_g + 1e-6)
    arr[:,:,2] *= avg_gray / (avg_b + 1e-6)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def invert_colors_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Invert image colors (for recovering from invert augmentation)."""
    return ImageOps.invert(image.convert('RGB'))


COLOR_RECOVERY_TOOLS = [
    RecoveryTool(
        name="auto_contrast",
        display_name="Auto Contrast",
        description="Automatically adjusts contrast by stretching histogram to use full range. Use when image looks washed out, low contrast, or hazy.",
        category="color",
        function=auto_contrast_tool,
        parameters={
            "cutoff": {
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 10,
                "description": "Percentage of pixels to cut from histogram ends"
            }
        },
        counters=["contrast", "fog", "frost"]
    ),
    RecoveryTool(
        name="gamma_correction",
        display_name="Gamma Correction",
        description="Adjusts overall brightness using gamma curve. gamma < 1 brightens dark images, gamma > 1 darkens bright images. Use for exposure correction.",
        category="color",
        function=gamma_correction_tool,
        parameters={
            "gamma": {
                "type": "float",
                "default": 1.0,
                "min": 0.3,
                "max": 3.0,
                "description": "Gamma value (< 1 brightens, > 1 darkens)"
            }
        },
        counters=["gamma", "gamma_up", "brightness", "brightness_up"]
    ),
    RecoveryTool(
        name="brightness_adjust",
        display_name="Brightness Adjustment",
        description="Directly adjust image brightness. Use when image is too dark or too bright.",
        category="color",
        function=brightness_adjust_tool,
        parameters={
            "factor": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "description": "Brightness factor (< 1 darkens, > 1 brightens)"
            }
        },
        counters=["brightness", "brightness_up"]
    ),
    RecoveryTool(
        name="contrast_adjust",
        display_name="Contrast Adjustment",
        description="Directly adjust image contrast. Use when image lacks contrast or has too much contrast.",
        category="color",
        function=contrast_adjust_tool,
        parameters={
            "factor": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "description": "Contrast factor (< 1 reduces, > 1 increases)"
            }
        },
        counters=["contrast", "contrast_up"]
    ),
    RecoveryTool(
        name="saturation_adjust",
        display_name="Saturation Adjustment",
        description="Increase or decrease color intensity. Use when colors appear too dull (desaturated) or too vivid (oversaturated).",
        category="color",
        function=saturation_adjust_tool,
        parameters={
            "factor": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 3.0,
                "description": "Saturation factor (0=grayscale, 1=unchanged, >1=more vivid)"
            }
        },
        counters=["saturation", "saturation_up", "grayscale"]
    ),
    RecoveryTool(
        name="histogram_equalize",
        display_name="Histogram Equalization",
        description="Spreads pixel intensities across full range for maximum contrast. Use for images with poor overall contrast distribution or posterization artifacts.",
        category="color",
        function=histogram_equalize_tool,
        parameters={},
        counters=["contrast", "posterize"]
    ),
    RecoveryTool(
        name="white_balance",
        display_name="White Balance",
        description="Corrects color cast using gray world assumption. Use when image has unnatural color tint (too warm, too cool, or color shifted).",
        category="color",
        function=white_balance_tool,
        parameters={},
        counters=["hue_shift", "color_jitter"]
    ),
    RecoveryTool(
        name="invert_colors",
        display_name="Invert Colors",
        description="Invert all colors in the image. Use only when image appears to be a negative/inverted version.",
        category="color",
        function=invert_colors_tool,
        parameters={},
        counters=["invert"]
    ),
]


# =============================================================================
# GEOMETRIC RECOVERY TOOLS
# =============================================================================

def rotate_correction_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Rotate image to correct misalignment."""
    degrees = params.get('degrees', 0)
    expand = params.get('expand', False)
    return image.convert('RGB').rotate(-degrees, expand=expand, fillcolor=(255, 255, 255), resample=Image.BILINEAR)


def flip_horizontal_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Flip image horizontally."""
    return ImageOps.mirror(image.convert('RGB'))


def flip_vertical_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Flip image vertically."""
    return ImageOps.flip(image.convert('RGB'))


def crop_borders_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Crop borders from image edges."""
    pixels = params.get('pixels', 10)
    img = image.convert('RGB')
    w, h = img.size

    # Ensure we don't crop more than image size
    pixels = min(pixels, min(w, h) // 4)

    if pixels <= 0:
        return img

    return img.crop((pixels, pixels, w - pixels, h - pixels))


def auto_crop_borders_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Automatically detect and crop solid color borders."""
    threshold = params.get('threshold', 10)
    arr = np.array(image.convert('RGB'))

    # Convert to grayscale for border detection
    gray = np.mean(arr, axis=2)

    # Get edge pixel values
    top_row = gray[0, :]
    bottom_row = gray[-1, :]
    left_col = gray[:, 0]
    right_col = gray[:, -1]

    # Detect if edges are uniform (border)
    def find_border_size(edge_values, full_dimension, axis_slice):
        """Find how many rows/cols are part of the border."""
        edge_mean = np.mean(edge_values)
        border_size = 0
        for i in range(min(full_dimension // 4, 100)):  # Don't crop more than 25%
            if axis_slice(i) is not None:
                row_or_col = axis_slice(i)
                if np.abs(np.mean(row_or_col) - edge_mean) < threshold:
                    border_size = i + 1
                else:
                    break
        return border_size

    h, w = gray.shape
    top_border = find_border_size(top_row, h, lambda i: gray[i, :])
    bottom_border = find_border_size(bottom_row, h, lambda i: gray[-(i+1), :])
    left_border = find_border_size(left_col, w, lambda i: gray[:, i])
    right_border = find_border_size(right_col, w, lambda i: gray[:, -(i+1)])

    # Crop if borders detected
    if top_border + bottom_border + left_border + right_border > 0:
        return image.crop((
            left_border,
            top_border,
            w - right_border,
            h - bottom_border
        ))

    return image


GEOMETRIC_RECOVERY_TOOLS = [
    RecoveryTool(
        name="rotate_correction",
        display_name="Rotation Correction",
        description="Rotate image to correct tilt or rotation. Specify degrees to rotate counter-clockwise (negative for clockwise).",
        category="geometric",
        function=rotate_correction_tool,
        parameters={
            "degrees": {
                "type": "float",
                "default": 0,
                "min": -45,
                "max": 45,
                "description": "Degrees to rotate counter-clockwise"
            },
            "expand": {
                "type": "bool",
                "default": False,
                "description": "Expand canvas to fit rotated image"
            }
        },
        counters=["rotate", "affine"]
    ),
    RecoveryTool(
        name="flip_horizontal",
        display_name="Horizontal Flip",
        description="Mirror the image horizontally (left-right). Use when image appears flipped.",
        category="geometric",
        function=flip_horizontal_tool,
        parameters={},
        counters=["flip_h"]
    ),
    RecoveryTool(
        name="flip_vertical",
        display_name="Vertical Flip",
        description="Flip the image vertically (upside-down). Use when image appears upside down.",
        category="geometric",
        function=flip_vertical_tool,
        parameters={},
        counters=["flip_v"]
    ),
    RecoveryTool(
        name="crop_borders",
        display_name="Crop Borders",
        description="Remove a fixed number of pixels from all edges. Use when image has visible borders.",
        category="geometric",
        function=crop_borders_tool,
        parameters={
            "pixels": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "description": "Number of pixels to crop from each edge"
            }
        },
        counters=["add_border"]
    ),
    RecoveryTool(
        name="auto_crop_borders",
        display_name="Auto Crop Borders",
        description="Automatically detect and remove solid color borders from image edges.",
        category="geometric",
        function=auto_crop_borders_tool,
        parameters={
            "threshold": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 50,
                "description": "Brightness tolerance for border detection"
            }
        },
        counters=["add_border"]
    ),
]


# =============================================================================
# WEATHER RECOVERY TOOLS
# =============================================================================

def dehaze_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Simple dehazing using contrast enhancement approach."""
    strength = params.get('strength', 0.5)

    arr = np.array(image.convert('RGB'), dtype=np.float32) / 255.0

    # Simple dehaze: increase contrast by stretching away from mean
    mean_val = np.mean(arr)
    dehazed = (arr - mean_val * strength) / (1 - strength + 0.01)
    dehazed = np.clip(dehazed, 0, 1)

    # Slight saturation boost to counter fog's desaturation
    gray = np.mean(dehazed, axis=2, keepdims=True)
    dehazed = dehazed + (dehazed - gray) * 0.2

    return Image.fromarray((np.clip(dehazed, 0, 1) * 255).astype(np.uint8))


def remove_rain_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Attempt to reduce rain streaks using median filtering."""
    strength = params.get('strength', 0.5)

    arr = np.array(image.convert('RGB'), dtype=np.float32)

    # Apply median filter to reduce streak patterns
    filtered = np.stack([median_filter(arr[:,:,c], size=3) for c in range(3)], axis=-1)

    # Blend original with filtered based on strength
    result = arr * (1 - strength) + filtered * strength

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def defrost_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Attempt to reduce frost overlay effects."""
    # Frost typically adds white overlay patterns
    # We can try to increase contrast and saturation
    strength = params.get('strength', 0.5)

    img = image.convert('RGB')

    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(1 + 0.5 * strength)

    # Increase saturation (frost desaturates)
    img = ImageEnhance.Color(img).enhance(1 + 0.3 * strength)

    # Slight auto-contrast
    img = ImageOps.autocontrast(img, cutoff=int(2 * strength))

    return img


WEATHER_RECOVERY_TOOLS = [
    RecoveryTool(
        name="dehaze",
        display_name="Dehaze/Defog",
        description="Remove haze or fog from image. Use when image appears washed out with reduced visibility and low contrast due to atmospheric effects.",
        category="weather",
        function=dehaze_tool,
        parameters={
            "strength": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 0.9,
                "description": "Dehazing strength (higher = more aggressive)"
            }
        },
        counters=["fog"]
    ),
    RecoveryTool(
        name="remove_rain",
        display_name="Remove Rain",
        description="Attempt to reduce visible rain streaks in image. Works best for light to moderate rain effects.",
        category="weather",
        function=remove_rain_tool,
        parameters={
            "strength": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 0.9,
                "description": "Removal strength"
            }
        },
        counters=["rain", "spatter"]
    ),
    RecoveryTool(
        name="defrost",
        display_name="Defrost",
        description="Reduce frost/ice overlay effects. Use when image has crystalline frost patterns reducing visibility.",
        category="weather",
        function=defrost_tool,
        parameters={
            "strength": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 1.0,
                "description": "Defrost strength"
            }
        },
        counters=["frost", "snow"]
    ),
]


# =============================================================================
# COMPRESSION RECOVERY TOOLS
# =============================================================================

def deblock_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Reduce JPEG blocking artifacts with light smoothing."""
    strength = params.get('strength', 0.5)

    arr = np.array(image.convert('RGB'), dtype=np.float32)

    # Light Gaussian blur to smooth block boundaries
    sigma = strength * 0.8  # Keep it subtle
    if len(arr.shape) == 3:
        blurred = np.stack([gaussian_filter(arr[:,:,c], sigma=sigma) for c in range(3)], axis=-1)
    else:
        blurred = gaussian_filter(arr, sigma=sigma)

    # Blend: keep more of original to preserve detail
    result = arr * 0.7 + blurred * 0.3

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def depixelate_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Smooth pixelated images using upscale-downscale approach."""
    scale = params.get('scale', 2)

    img = image.convert('RGB')
    w, h = img.size

    # Upscale with bilinear to smooth pixels
    upscaled = img.resize((w * scale, h * scale), Image.BILINEAR)

    # Downscale with Lanczos for quality
    return upscaled.resize((w, h), Image.LANCZOS)


COMPRESSION_RECOVERY_TOOLS = [
    RecoveryTool(
        name="deblock",
        display_name="JPEG Deblocking",
        description="Reduce JPEG compression blocking artifacts (visible square block patterns, especially in smooth areas like sky or skin).",
        category="compression",
        function=deblock_tool,
        parameters={
            "strength": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 1.0,
                "description": "Deblocking strength"
            }
        },
        counters=["jpeg_compression"]
    ),
    RecoveryTool(
        name="depixelate",
        display_name="Depixelate",
        description="Smooth pixelated/blocky images to appear more natural. Use when image has visible large square pixels.",
        category="compression",
        function=depixelate_tool,
        parameters={
            "scale": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 4,
                "description": "Upscale factor for smoothing"
            }
        },
        counters=["pixelate", "downsample"]
    ),
]


# =============================================================================
# RESOLUTION RECOVERY TOOLS
# =============================================================================

def upscale_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Upscale image using Lanczos interpolation."""
    scale = params.get('scale', 2.0)

    img = image.convert('RGB')
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)

    return img.resize((new_w, new_h), Image.LANCZOS)


def zoom_crop_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Crop center of image and upscale to original size (zoom effect)."""
    zoom_factor = params.get('zoom_factor', 1.5)

    img = image.convert('RGB')
    w, h = img.size

    # Calculate crop box
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2

    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def smart_crop_tool(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Crop to focus on a specific region."""
    region = params.get('region', 'center')  # center, top, bottom, left, right
    crop_ratio = params.get('crop_ratio', 0.8)  # Keep 80% of image

    img = image.convert('RGB')
    w, h = img.size

    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    if region == 'center':
        left = (w - new_w) // 2
        top = (h - new_h) // 2
    elif region == 'top':
        left = (w - new_w) // 2
        top = 0
    elif region == 'bottom':
        left = (w - new_w) // 2
        top = h - new_h
    elif region == 'left':
        left = 0
        top = (h - new_h) // 2
    elif region == 'right':
        left = w - new_w
        top = (h - new_h) // 2
    else:
        left = (w - new_w) // 2
        top = (h - new_h) // 2

    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


RESOLUTION_RECOVERY_TOOLS = [
    RecoveryTool(
        name="upscale",
        display_name="Upscale",
        description="Increase image resolution using high-quality Lanczos interpolation. Use for small or low-resolution images that need more detail.",
        category="resolution",
        function=upscale_tool,
        parameters={
            "scale": {
                "type": "float",
                "default": 2.0,
                "min": 1.5,
                "max": 4.0,
                "description": "Upscale factor"
            }
        },
        counters=["downsample"]
    ),
    RecoveryTool(
        name="zoom_crop",
        display_name="Zoom & Crop",
        description="Zoom into the center of the image by cropping edges and upscaling. Use to focus on central content and eliminate distracting borders or peripheral noise.",
        category="resolution",
        function=zoom_crop_tool,
        parameters={
            "zoom_factor": {
                "type": "float",
                "default": 1.5,
                "min": 1.1,
                "max": 3.0,
                "description": "Zoom factor (1.5 = 50% zoom in)"
            }
        },
        counters=["text_overlay", "watermark"]  # Can help by zooming past overlays
    ),
    RecoveryTool(
        name="smart_crop",
        display_name="Smart Crop",
        description="Crop to focus on a specific region of the image. Use when important content is in a particular area and you want to remove distractions.",
        category="resolution",
        function=smart_crop_tool,
        parameters={
            "region": {
                "type": "str",
                "default": "center",
                "options": ["center", "top", "bottom", "left", "right"],
                "description": "Region to focus on"
            },
            "crop_ratio": {
                "type": "float",
                "default": 0.8,
                "min": 0.5,
                "max": 0.95,
                "description": "Portion of image to keep"
            }
        },
        counters=["random_occlusion", "grid_mask"]
    ),
]


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Central registry for all recovery tools."""

    def __init__(self):
        self.tools: Dict[str, RecoveryTool] = {}
        self.categories: Dict[str, List[str]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all built-in tools."""
        all_tools = (
            BLUR_RECOVERY_TOOLS +
            NOISE_RECOVERY_TOOLS +
            COLOR_RECOVERY_TOOLS +
            GEOMETRIC_RECOVERY_TOOLS +
            WEATHER_RECOVERY_TOOLS +
            COMPRESSION_RECOVERY_TOOLS +
            RESOLUTION_RECOVERY_TOOLS
        )
        for tool in all_tools:
            self.register(tool)

    def register(self, tool: RecoveryTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        if tool.name not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.name)

    def get(self, name: str) -> Optional[RecoveryTool]:
        """Get tool by name."""
        return self.tools.get(name)

    def get_by_category(self, category: str) -> List[RecoveryTool]:
        """Get all tools in a category."""
        return [self.tools[name] for name in self.categories.get(category, [])]

    def get_tools_for_corruption(self, corruption_name: str) -> List[RecoveryTool]:
        """Get tools that can counter a specific corruption."""
        return [tool for tool in self.tools.values() if corruption_name in tool.counters]

    def get_all_tools(self) -> List[RecoveryTool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_tool_names(self) -> List[str]:
        """Get all tool names."""
        return list(self.tools.keys())

    def get_tool_descriptions_for_vlm(self) -> str:
        """Format all tool descriptions for VLM prompt."""
        lines = ["Available Recovery Tools:\n"]

        for category in sorted(self.categories.keys()):
            lines.append(f"\n## {category.upper()}")
            for tool_name in self.categories[category]:
                tool = self.tools[tool_name]
                lines.append(f"\n### {tool.name}")
                lines.append(f"**Description:** {tool.description}")

                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param_name, param_info in tool.parameters.items():
                        param_type = param_info.get('type', 'any')
                        default = param_info.get('default', 'N/A')
                        desc = param_info.get('description', '')
                        lines.append(f"  - `{param_name}` ({param_type}, default={default}): {desc}")
                else:
                    lines.append("**Parameters:** None")

        return "\n".join(lines)

    def get_tool_list_short(self) -> str:
        """Get a short list of tools for prompts."""
        lines = []
        for category in sorted(self.categories.keys()):
            tool_names = self.categories[category]
            lines.append(f"- {category}: {', '.join(tool_names)}")
        return "\n".join(lines)


# Global registry instance
TOOL_REGISTRY = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return TOOL_REGISTRY


# =============================================================================
# IMAGE QUALITY METRICS
# =============================================================================

def measure_sharpness(image: Image.Image) -> float:
    """
    Measure image sharpness using Laplacian variance.
    Higher values = sharper image.

    Reference values:
    - Very blurry: < 100
    - Blurry: 100-500
    - Moderate: 500-2000
    - Sharp: 2000-5000
    - Very sharp: > 5000
    """
    gray = np.array(image.convert('L'), dtype=np.float64)
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.ndimage import convolve
    edges = convolve(gray, laplacian)
    return float(np.var(edges))


def measure_contrast(image: Image.Image) -> dict:
    """
    Measure image contrast using multiple methods.

    Returns dict with:
    - pixel_range: max - min pixel value (0-255 scale)
    - rms_contrast: RMS contrast (std / mean)
    - percentile_range: 95th - 5th percentile (robust to outliers)

    Reference values for pixel_range:
    - Very low contrast: < 50
    - Low contrast: 50-100
    - Normal: 100-200
    - High contrast: > 200
    """
    arr = np.array(image.convert('L'), dtype=np.float64)

    pixel_range = float(arr.max() - arr.min())
    rms_contrast = float(arr.std() / (arr.mean() + 1e-6))
    percentile_range = float(np.percentile(arr, 95) - np.percentile(arr, 5))

    return {
        "pixel_range": round(pixel_range, 1),
        "rms_contrast": round(rms_contrast, 3),
        "percentile_range": round(percentile_range, 1)
    }


def measure_brightness(image: Image.Image) -> dict:
    """
    Measure image brightness.

    Returns dict with:
    - mean: mean pixel value (0-255)
    - median: median pixel value
    - underexposed_pct: % of pixels below 30
    - overexposed_pct: % of pixels above 225

    Reference values for mean:
    - Very dark: < 50
    - Dark: 50-80
    - Normal: 80-180
    - Bright: 180-220
    - Very bright: > 220
    """
    arr = np.array(image.convert('L'), dtype=np.float64)

    return {
        "mean": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "underexposed_pct": round(float((arr < 30).sum() / arr.size * 100), 1),
        "overexposed_pct": round(float((arr > 225).sum() / arr.size * 100), 1)
    }


def measure_noise(image: Image.Image) -> float:
    """
    Estimate noise level using median absolute deviation of Laplacian.
    Higher values = more noise.

    Reference values:
    - Clean: < 5
    - Low noise: 5-15
    - Moderate noise: 15-30
    - High noise: 30-50
    - Very noisy: > 50
    """
    gray = np.array(image.convert('L'), dtype=np.float64)
    # Laplacian
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.ndimage import convolve
    edges = convolve(gray, laplacian)
    # Median absolute deviation (robust noise estimate)
    sigma = np.median(np.abs(edges)) / 0.6745
    return round(float(sigma), 2)


def measure_saturation(image: Image.Image) -> dict:
    """
    Measure color saturation.

    Returns dict with:
    - mean: mean saturation (0-1 scale)
    - std: saturation standard deviation
    - low_sat_pct: % of pixels with saturation < 0.1

    Reference values for mean:
    - Grayscale/desaturated: < 0.1
    - Low saturation: 0.1-0.3
    - Normal: 0.3-0.6
    - High saturation: > 0.6
    """
    rgb = np.array(image.convert('RGB'), dtype=np.float64) / 255.0

    # Compute saturation (HSV style)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    delta = max_rgb - min_rgb

    # Saturation = delta / max (avoid division by zero)
    saturation = np.where(max_rgb > 0, delta / (max_rgb + 1e-6), 0)

    return {
        "mean": round(float(saturation.mean()), 3),
        "std": round(float(saturation.std()), 3),
        "low_sat_pct": round(float((saturation < 0.1).sum() / saturation.size * 100), 1)
    }


def measure_colorfulness(image: Image.Image) -> float:
    """
    Measure colorfulness using Hasler and Süsstrunk's metric.
    Higher values = more colorful.

    Reference values:
    - Grayscale: ~0
    - Not colorful: < 20
    - Slightly colorful: 20-40
    - Moderately colorful: 40-70
    - Colorful: 70-100
    - Very colorful: > 100
    """
    rgb = np.array(image.convert('RGB'), dtype=np.float64)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # Compute rg and yb
    rg = R - G
    yb = 0.5 * (R + G) - B

    # Compute mean and std
    rg_mean, rg_std = rg.mean(), rg.std()
    yb_mean, yb_std = yb.mean(), yb.std()

    # Colorfulness metric
    std_root = np.sqrt(rg_std**2 + yb_std**2)
    mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
    colorfulness = std_root + 0.3 * mean_root

    return round(float(colorfulness), 2)


def measure_jpeg_artifacts(image: Image.Image) -> float:
    """
    Estimate JPEG blocking artifacts.
    Higher values = more blocking artifacts.

    Reference values:
    - No artifacts: < 5
    - Mild artifacts: 5-15
    - Moderate artifacts: 15-30
    - Severe artifacts: > 30
    """
    gray = np.array(image.convert('L'), dtype=np.float64)
    h, w = gray.shape

    # Check for 8x8 block boundaries (JPEG uses 8x8 DCT blocks)
    block_size = 8

    # Compute differences at block boundaries vs within blocks
    boundary_diffs = []
    interior_diffs = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            diff = abs(gray[i, j] - gray[i, j-1])
            if j % block_size == 0:
                boundary_diffs.append(diff)
            else:
                interior_diffs.append(diff)

    if not boundary_diffs or not interior_diffs:
        return 0.0

    # Blocking artifact metric: ratio of boundary to interior differences
    boundary_mean = np.mean(boundary_diffs)
    interior_mean = np.mean(interior_diffs)

    if interior_mean < 1e-6:
        return 0.0

    # Higher ratio = more blocking
    artifact_score = (boundary_mean / (interior_mean + 1e-6) - 1) * 100
    return round(max(0, float(artifact_score)), 2)


def get_all_metrics(image: Image.Image) -> dict:
    """
    Compute all image quality metrics.

    Returns a dictionary with all metrics that can be passed to VLM
    for informed decision making.
    """
    return {
        "sharpness": measure_sharpness(image),
        "contrast": measure_contrast(image),
        "brightness": measure_brightness(image),
        "noise": measure_noise(image),
        "saturation": measure_saturation(image),
        "colorfulness": measure_colorfulness(image),
        "jpeg_artifacts": measure_jpeg_artifacts(image)
    }


def format_metrics_for_vlm(metrics: dict) -> str:
    """
    Format metrics as a readable string for VLM.
    Includes reference values for context.
    """
    lines = [
        "## Image Quality Metrics\n",
        f"**Sharpness:** {metrics['sharpness']:.1f}",
        "  (Reference: <100=very blurry, 100-500=blurry, 500-2000=moderate, >2000=sharp)\n",
        f"**Contrast:**",
        f"  - Pixel range: {metrics['contrast']['pixel_range']}",
        f"  - RMS contrast: {metrics['contrast']['rms_contrast']}",
        f"  - Percentile range (5-95): {metrics['contrast']['percentile_range']}",
        "  (Reference pixel_range: <50=very low, 50-100=low, 100-200=normal, >200=high)\n",
        f"**Brightness:**",
        f"  - Mean: {metrics['brightness']['mean']}",
        f"  - Underexposed: {metrics['brightness']['underexposed_pct']}%",
        f"  - Overexposed: {metrics['brightness']['overexposed_pct']}%",
        "  (Reference mean: <50=very dark, 50-80=dark, 80-180=normal, >180=bright)\n",
        f"**Noise level:** {metrics['noise']}",
        "  (Reference: <5=clean, 5-15=low, 15-30=moderate, >30=high)\n",
        f"**Saturation:**",
        f"  - Mean: {metrics['saturation']['mean']}",
        f"  - Low saturation pixels: {metrics['saturation']['low_sat_pct']}%",
        "  (Reference mean: <0.1=desaturated, 0.1-0.3=low, 0.3-0.6=normal, >0.6=high)\n",
        f"**Colorfulness:** {metrics['colorfulness']}",
        "  (Reference: <20=not colorful, 20-40=slightly, 40-70=moderate, >70=colorful)\n",
        f"**JPEG artifacts:** {metrics['jpeg_artifacts']}",
        "  (Reference: <5=none, 5-15=mild, 15-30=moderate, >30=severe)"
    ]
    return "\n".join(lines)
