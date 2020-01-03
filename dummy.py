import argparse
import colorsys
import math

import matplotlib
import numpy as np
from IPython import embed
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate

W = H = 2
x = np.arange(W)
y = np.arange(H)
noise = np.array([[0.2, 0.4], [0.6, 0.8]])

f = interpolate.interp2d(x, y, noise)
from IPython import embed; embed(); exit();
x_test = np.array([0.5])
y_test = np.array([0.5])

print(f(x_test, y_test))
