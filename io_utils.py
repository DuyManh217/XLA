import numpy as np
from PIL import Image


def read_image(path: str, is_gray: bool = True) -> np.ndarray:
    """Đọc ảnh và trả về mảng NumPy."""
    with Image.open(path) as image:
        image = image.convert("L" if is_gray else "RGB")
        return np.asarray(image).copy()


def save_image(path: str, image: np.ndarray) -> None:
    """Ghi mảng NumPy ra file ảnh."""
    output = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(output).save(path)

    # image sau khi đọc từ cv2.imread là ma trận BGR


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def bgr_to_gray(image):
    # Ép kiểu sang float32 để tránh tràn số khi nhân
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # Áp dụng công thức chuẩn
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Ép lại về uint8 (0-255)
    return np.clip(gray, 0, 255).astype(np.uint8)


def order_points(pts):
    """Return four cyclic corners, starting with the topmost (then leftmost).

    Unlike independent sum/difference argmins this cannot duplicate a corner
    for diamonds or boxes rotated exactly 45 degrees.
    """
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    ordered = points[np.argsort(angles)]
    first = np.lexsort((ordered[:, 0], ordered[:, 1]))[0]
    return np.roll(ordered, -first, axis=0)
