from matplotlib.colors import hsv_to_rgb
import numpy as np

def generate_colors(num_colors):
    hues = np.linspace(0, 1, num_colors, endpoint=False)  # Evenly spaced hues
    saturation = 0.9  # High saturation
    brightness = 0.9  # High brightness
    colors = [hsv_to_rgb([hue, saturation, brightness]) for hue in hues]
    return (np.array(colors) * 255).astype(np.uint8)  # Convert to 8-bit RGB