import numpy as np


def _validate_inputs(image: np.ndarray, kernel: np.ndarray) -> tuple[int, int]:
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("ảnh và kernel phải là mảng 2 chiều")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel phải có kích thước lẻ")
    return kernel.shape


def dilation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Phép giãn hình thái học."""
    kh, kw = _validate_inputs(image, kernel)
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    result = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            region = padded[row : row + kh, column : column + kw]
            if np.any((region == 255) & (kernel == 1)):
                result[row, column] = 255
    return result


def erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Phép co hình thái học."""
    kh, kw = _validate_inputs(image, kernel)
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    result = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            region = padded[row : row + kh, column : column + kw]
            if np.all((region == 255) | (kernel == 0)):
                result[row, column] = 255
    return result


def opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Phép mở = co rồi giãn."""
    return dilation(erosion(image, kernel), kernel)


def closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Phép đóng = giãn rồi co."""
    return erosion(dilation(image, kernel), kernel)


def extract_boundary(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tách biên theo công thức beta(A) = A - (A co B)."""
    eroded = erosion(image, kernel)
    boundary = np.zeros_like(image)
    boundary[(image == 255) & (eroded == 0)] = 255
    return boundary
