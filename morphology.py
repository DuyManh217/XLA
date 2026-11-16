"""Binary morphology implemented using NumPy and active kernel offsets."""
from collections import deque
import numpy as np


def _morph(image, kernel, dilate):
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError('Ảnh và kernel phải có 2 chiều')
    kh, kw = kernel.shape
    if kh == 0 or kw == 0 or kh % 2 == 0 or kw % 2 == 0 or not np.any(kernel):
        raise ValueError('Kernel phải lẻ dương và có phần tử hoạt động')
    h, w = image.shape
    padded = np.pad(image > 0, ((kh//2, kh//2), (kw//2, kw//2)), constant_values=False)
    result = np.zeros((h, w), bool) if dilate else np.ones((h, w), bool)
    for y, x in np.argwhere(kernel != 0):
        region = padded[y:y+h, x:x+w]
        if dilate:
            result |= region
        else:
            result &= region
    return result.astype(np.uint8) * 255


def dilation(image, kernel):
    return _morph(image, kernel, True)


def erosion(image, kernel):
    return _morph(image, kernel, False)


def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)


def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)


def extract_boundary(image, kernel):
    return ((image > 0) & (erosion(image, kernel) == 0)).astype(np.uint8) * 255


def fill_holes(image):
    """Fill black regions enclosed by white pixels; 4-neighbor background flood fill."""
    h, w = image.shape
    background = image == 0
    outside = np.zeros((h, w), bool)
    queue = deque()
    for y in range(h):
        for x in (0, w - 1):
            if background[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((x, y))
    for x in range(w):
        for y in (0, h - 1):
            if background[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and background[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue.append((nx, ny))
    result = image.copy()
    result[background & ~outside] = 255
    return result
