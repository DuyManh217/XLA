import numpy as np
from PIL import Image


def read_image(path: str, is_gray: bool = True) -> np.ndarray:
    """Đọc ảnh và trả về mảng NumPy."""
    image = Image.open(path)
    if is_gray:
        image = image.convert("L")
    return np.asarray(image)


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
    """
    Sắp xếp 4 điểm: Trên-Trái (TL), Trên-Phải (TR), Dưới-Phải (BR), Dưới-Trái (BL).
    :param pts: Mảng 4 điểm (shape: 4x2)
    """
    # Khởi tạo mảng kết quả
    rect = np.zeros((4, 2), dtype="float32")

    # 1. Tính tổng (x + y)
    s = pts.sum(axis=1)
    # Điểm có tổng nhỏ nhất là Top-Left
    rect[0] = pts[np.argmin(s)]
    # Điểm có tổng lớn nhất là Bottom-Right
    rect[2] = pts[np.argmax(s)]

    # 2. Tính hiệu (x - y)
    diff = np.diff(pts, axis=1)
    # Điểm có hiệu nhỏ nhất là Top-Right
    rect[1] = pts[np.argmin(diff)]
    # Điểm có hiệu lớn nhất là Bottom-Left
    rect[3] = pts[np.argmax(diff)]

    return rect
