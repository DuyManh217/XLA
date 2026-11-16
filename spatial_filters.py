"""NumPy spatial filters; vectorized over pixels, no cv2 filtering calls."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def convolve_2d(image, kernel, padding='constant'):
    """Kernel correlation (legacy API name); symmetric Gaussian = convolution."""
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError('Ảnh và kernel phải có 2 chiều')
    kh, kw = kernel.shape
    if not kh or not kw or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('Kernel phải có kích thước lẻ dương')
    padded = np.pad(image.astype(np.float64), ((kh//2, kh//2), (kw//2, kw//2)), mode=padding)
    result = np.zeros(image.shape, dtype=np.float64)
    # No H*W*K*K temporary allocation.
    for y in range(kh):
        for x in range(kw):
            result += padded[y:y+image.shape[0], x:x+image.shape[1]] * kernel[y, x]
    return result


def median_filter(image, ksize=3):
    if image.ndim != 2 or ksize <= 0 or ksize % 2 == 0:
        raise ValueError('Ảnh 2 chiều và ksize lẻ dương là bắt buộc')
    padded = np.pad(image, ksize//2, mode='edge')
    return np.median(sliding_window_view(padded, (ksize, ksize)), axis=(-2, -1)).astype(image.dtype)


def create_gaussian_kernel(ksize, sigma=0):
    if ksize <= 0 or ksize % 2 == 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('ksize phải lẻ dương, sigma phải hữu hạn và không âm')
    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(-(ksize//2), ksize//2+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(img, ksize=7, sigma=0):
    return np.clip(convolve_2d(img, create_gaussian_kernel(ksize, sigma), 'reflect'), 0, 255).astype(np.uint8)
