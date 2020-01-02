import argparse
import colorsys
import math

import matplotlib
import numpy as np
from IPython import embed
from PIL import Image
from scipy import interpolate


def save_image(arr: np.ndarray, name: str):
    """
        Helper to save an image from numpy array
    """
    assert arr.dtype == np.float64
    assert arr.max() <= 1.0 and arr.min() >= 0.0
    image = Image.fromarray((arr * 255.0).astype(np.uint8))
    image.save(name)


def generate_noise(len_canvas: int):
    size = (len_canvas, len_canvas)
    INT_MAX = 32768
    noise = np.random.randint(INT_MAX, size=size) / (INT_MAX - 1)
    return noise


def turbulence(noise: np.ndarray, turb_size: int):
    """ Integrate multi-scale noise
    Arguments:
        noise (np.ndarray): 2-dim. numpy array with range [0.0, 1.0]
        turb_size {int}: bigger turb_size produces smoother texture
    Returns:
        noise_multi_scale (np.ndarray): 2-dim. numpy array with range [0.0, 1.0]
    """
    assert noise.ndim == 2
    H, W = noise.shape
    x = np.arange(W)
    y = np.arange(H)
    f = interpolate.interp2d(x, y, noise)

    size = turb_size
    value = np.zeros(shape=noise.shape)

    while True:
        # adding log_{2}^{256} + 1 multi-scale noise
        # weight for low-frequency noise is higher than high-frequency noise
        value += f(x / size, y / size) * size
        size = size // 2
        if size == 0:
            break
    # x + x / 2 + x / 4 + ... = 2x, so normalize it by 2x
    noise_multi_scale = value / (turb_size * 2)
    return noise_multi_scale


def generate_cloud(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    noise = generate_noise(args.len_canvas)
    noise_multi_scale = turbulence(noise, 64)
    save_image(noise_multi_scale, 'noise_multi_scale.png')

    hue = np.full(noise.shape, 169 / 255)
    saturation = np.full(noise.shape, 255 / 255)
    lightness = np.clip((192 / 255 + noise_multi_scale / 4),
                        a_min=0.0, a_max=1.0)
    hls_arr = np.stack([hue, lightness, saturation], axis=-1)
    rgb_arr = np.empty_like(hls_arr)

    for i in range(args.len_canvas):
        for j in range(args.len_canvas):
            # TODO: vectorize this operation
            rgb_arr[i][j] = np.array(colorsys.hls_to_rgb(*hls_arr[i][j]))

    save_image(rgb_arr, 'cloud.png')


def generate_marble(parser: argparse.ArgumentParser):
    parser.add_argument('--period_x', type=float, default=5.0)
    parser.add_argument('--period_y', type=float, default=10.0)
    parser.add_argument('--turb_power', type=float, default=5.0)
    parser.add_argument('--turb_size', type=int, default=32)

    args = parser.parse_args()

    noise = generate_noise(args.len_canvas)
    noise_multi_scale = turbulence(noise, args.turb_size)

    H, W = noise.shape
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    def pointwise_func(x_coord, y_coord, turb):
        value_xy = x_coord * args.period_x / args.len_canvas
        value_xy = value_xy + y_coord * args.period_y / args.len_canvas
        value_xy = value_xy + args.turb_power * turb
        return np.abs(np.sin(value_xy * math.pi))

    vfunc = np.vectorize(pointwise_func)
    arr = vfunc(xx, yy, noise_multi_scale)
    save_image(arr, 'murble.png')


def generate_wood(parser: argparse.ArgumentParser):
    parser.add_argument('--period_xy', type=float, default=12.0)
    parser.add_argument('--turb_power', type=float, default=0.1)
    parser.add_argument('--turb_size', type=int, default=32)

    args = parser.parse_args()

    noise = generate_noise(args.len_canvas)
    noise_multi_scale = turbulence(noise, args.turb_size)

    H, W = noise.shape
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)  # shape: (W, H)

    def pointwise_func(x_coord, y_coord, turb, height, width):

        value_x = (x_coord - width / 2) / width
        value_y = (y_coord - height / 2) / height
        dist = np.sqrt(value_x ** 2 + value_y ** 2)
        dist = dist + args.turb_power * turb

        return np.abs(np.sin(2 * args.period_xy * dist * math.pi))

    vfunc = np.vectorize(pointwise_func)
    arr = vfunc(xx, yy, noise_multi_scale, H, W)
    save_image(arr, 'wood.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_canvas', type=int, default=128)
    generate_wood(parser)
        # from IPython import embed; embed(); exit();
