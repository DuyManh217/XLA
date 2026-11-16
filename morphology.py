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


def fill_holes(binary_img: np.ndarray) -> np.ndarray:
    """
    Lấp đầy tất cả các lỗ thủng bên trong vật thể (Hole filling).
    Thuật toán: Loang BFS từ 4 đường viền ảnh trên mặt nạ nền bù,
    các điểm nền không chạm tới viền chính là lỗ thủng bên trong đối tượng.
    """
    h, w = binary_img.shape
    inv_mask = (binary_img == 0)  # True là nền, False là vật thể

    visited = np.zeros((h, w), dtype=bool)
    queue = []

    # Thêm các điểm nền ở 4 cạnh viền ảnh vào hàng đợi
    for x in range(w):
        if inv_mask[0, x] and not visited[0, x]:
            visited[0, x] = True
            queue.append((x, 0))
        if inv_mask[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            queue.append((x, h - 1))

    for y in range(h):
        if inv_mask[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            queue.append((0, y))
        if inv_mask[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            queue.append((w - 1, y))

    # Loang BFS 4 hướng
    head = 0
    while head < len(queue):
        cx, cy = queue[head]
        head += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if inv_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    # Những điểm là nền (inv_mask) nhưng không được chạm tới từ viền ngoài chính là lỗ thủng
    result = binary_img.copy()
    holes = inv_mask & (~visited)
    result[holes] = 255
    return result
