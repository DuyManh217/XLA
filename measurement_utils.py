"""Các kiểm tra nhỏ dùng chung cho CLI, GUI và benchmark."""
import sys
from pathlib import Path
import cv2
import numpy as np
from contours import contour_area, min_area_rect, sort_contours


ROOT = Path(__file__).resolve().parent


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_bgr(path):
    """Hỗ trợ cả đường dẫn có dấu trên Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError("Không đọc được ảnh: " + str(path))
    return image


def save_bgr(path, image):
    ok, data = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise OSError("Không mã hóa được ảnh.")
    data.tofile(path)


def prepare_measurement(contours, width, reference=1):
    """Vật chuẩn đánh số từ 1, trái sang phải; width là cạnh dài/đường kính."""
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Kích thước vật chuẩn phải là số hữu hạn lớn hơn 0.")
    contours, _ = sort_contours([c for c in contours if contour_area(c) >= 100])
    if not contours:
        raise ValueError("Không tìm thấy vật thể. Thử nền tương phản, các vật tách rời.")
    if not 1 <= reference <= len(contours):
        raise ValueError(f"Số thứ tự vật chuẩn phải từ 1 đến {len(contours)}.")
    _, sides, _ = min_area_rect(contours[reference - 1])
    ppm = max(sides) / width
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Vật chuẩn không có kích thước hợp lệ.")
    return contours, ppm
