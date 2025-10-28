from matplotlib.colors import hsv_to_rgb
import numpy as np
import psutil

def generate_colors(num_colors):
    hues = np.linspace(0, 1, num_colors, endpoint=False)  # Evenly spaced hues
    saturation = 0.9  # High saturation
    brightness = 0.9  # High brightness
    colors = [hsv_to_rgb([hue, saturation, brightness]) for hue in hues]
    return (np.array(colors) * 255).astype(np.uint8)  # Convert to 8-bit RGB


# Convert data to JSON serializable format
def convert_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {k: convert_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(v) for v in data]
    else:
        return data

# Function to log resource utilization
def log_resource_usage(stage):
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"[{stage}] CPU Usage: {cpu_percent}%")
    print(f"[{stage}] Memory Usage: {memory.percent}% ({memory.used / (1024**2):.2f} MB used / {memory.total / (1024**2):.2f} MB total)")
