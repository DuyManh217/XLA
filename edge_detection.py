import cv2
import numpy as np
from collections import deque

from spatial_filters import convolve_2d


def canny_opencv(
    image: np.ndarray,
    low_threshold: float = 50,
    high_threshold: float = 100,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """Phát hiện biên bằng OpenCV Canny."""
    if image.ndim != 2:
        raise ValueError("canny_edge_detection yêu cầu ảnh 2 chiều")
    if low_threshold < 0 or high_threshold < low_threshold:
        raise ValueError("cần 0 <= low_threshold <= high_threshold")
    if aperture_size not in (3, 5, 7):
        raise ValueError("aperture_size phải là 3, 5 hoặc 7")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return cv2.Canny(
        image,
        low_threshold,
        high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )


def canny_edge_detection(image, low_threshold=50, high_threshold=100):
    """Canny tự cài: Sobel, làm mảnh biên, ngưỡng kép và nối biên.

    Ảnh đầu vào đã được làm mịn Gaussian ở chương trình chính.
    Dùng độ lớn L1 để gần với cấu hình Canny OpenCV mặc định.
    """
    if image.ndim != 2 or not 0 <= low_threshold <= high_threshold:
        raise ValueError("Cần ảnh xám và 0 <= ngưỡng thấp <= ngưỡng cao.")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = convolve_2d(image, kx)
    gy = convolve_2d(image, kx.T)
    magnitude = np.abs(gx) + np.abs(gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180
    thin = np.zeros_like(magnitude)
    h, w = image.shape
    # So sánh hai láng giềng theo hướng gradient.
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            if a < 22.5 or a >= 157.5:
                p, q = magnitude[y, x - 1], magnitude[y, x + 1]
            elif a < 67.5:
                p, q = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            elif a < 112.5:
                p, q = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                p, q = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            if magnitude[y, x] > p and magnitude[y, x] >= q:
                thin[y, x] = magnitude[y, x]
    result = np.zeros(image.shape, dtype=np.uint8)
    strong = (thin > 0) & (thin >= high_threshold)
    result[strong] = 255
    queue = deque(zip(*np.where(strong)))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if result[ny, nx] == 0 and thin[ny, nx] > 0 and thin[ny, nx] >= low_threshold:
                        result[ny, nx] = 255
                        queue.append((ny, nx))
    return result


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """Phát hiện biên bằng toán tử Sobel."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = convolve_2d(image, kernel_x)
    gradient_y = convolve_2d(image, kernel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
    """Phát hiện biên bằng toán tử Laplace."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    response = np.abs(convolve_2d(image, kernel))
    return np.clip(response, 0, 255).astype(np.uint8)
