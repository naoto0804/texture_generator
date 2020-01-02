from IPython import embed
import argparse
import matplotlib

import numpy as np
from PIL import Image
from scipy import interpolate

import colorsys

def save_image(arr: np.ndarray, name: str):
    """
        Helper to save an image from numpy array
    """
    assert arr.dtype == np.float64
    assert arr.max() <= 1.0 and arr.min() >= 0.0
    image = Image.fromarray((arr * 255.0).astype(np.uint8))
    image.save(name)

def generate_cloud(len_canvas):
    size = (len_canvas, len_canvas)
    INT_MAX = 32768
    noise = np.random.randint(INT_MAX, size=size) / (INT_MAX - 1)

    # Construct interpolation field
    x = y = np.arange(len_canvas)
    f = interpolate.interp2d(x, y, noise)

    size = initial_size = 64
    value = np.zeros(shape=noise.shape)
    while True:
        # adding log_{2}^{256} + 1 multi-scale noise
        # weight for low-frequency noise is higher than high-frequency noise
        value += f(x / size, y / size) * size
        size = size // 2
        if size == 0:
            break
    # x + x / 2 + x / 4 + ... = 2x, so normalize it by 2x
    noise_multi_scale = value / (initial_size * 2)
    save_image(noise_multi_scale, 'noise_multi_scale.png')

    hue = np.full(noise.shape, 169 / 255)
    saturation = np.full(noise.shape, 255 / 255)
    lightness = np.clip((192 / 255 + noise_multi_scale / 4), a_min=0.0, a_max=1.0)
    hls_arr = np.stack([hue, lightness, saturation], axis=-1)
    rgb_arr = np.empty_like(hls_arr)

    for i in range(len_canvas):
        for j in range(len_canvas):
            # TODO: vectorize this operation
            rgb_arr[i][j] = np.array(colorsys.hls_to_rgb(*hls_arr[i][j]))

    save_image(rgb_arr, 'cloud.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_canvas', type=int, default=128)
    args = parser.parse_args()
    generate_cloud(args.len_canvas)
