import cv2
import numpy as np

from spatial_filters import convolve_2d


def canny_edge_detection(
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
