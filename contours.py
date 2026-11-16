"""Ordered external contours. OpenCV is used explicitly for geometric correctness.

The old flood-fill boundary is an unordered point cloud: applying the polygon
shoelace formula to it gives invalid areas. Original code is in legacy_original.
"""
import cv2
import numpy as np


def find_contours(binary_img):
    if binary_img.ndim != 2:
        raise ValueError("Ảnh nhị phân phải có 2 chiều")
    binary = np.where(binary_img > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def sort_contours(cnts, method="left-to-right"):
    if not cnts:
        return [], np.empty((0, 4), dtype=int)
    if method not in ('left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'):
        raise ValueError("Hướng sắp xếp không hợp lệ")
    axis = 1 if method in ('top-to-bottom', 'bottom-to-top') else 0
    pairs = sorted(((c, cv2.boundingRect(c)) for c in cnts),
                   key=lambda p: (p[1][axis], p[1][1-axis]),
                   reverse=method in ('right-to-left', 'bottom-to-top'))
    return [p[0] for p in pairs], np.array([p[1] for p in pairs])


def contour_area(c):
    """Shoelace for an ORDERED polygon, float64 to avoid integer overflow."""
    pts = np.asarray(c, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts.T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def min_area_rect(c):
    return cv2.minAreaRect(np.asarray(c, dtype=np.float32))


def box_points(rect):
    # Keep subpixel precision until drawing.
    return cv2.boxPoints(rect)
