import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def render_density_map(density_map: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='jet', interpolation='nearest', alpha=0.9)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img
