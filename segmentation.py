"""Phân vùng có hướng dẫn: hình chữ nhật, nét tiền cảnh/nền hoặc đa giác."""
import cv2
import numpy as np


def grabcut_mask(rgb, selection):
    if not selection:
        raise ValueError('Hãy khoanh vùng quanh vật cần đo trước.')
    if selection.get('polygon'):
        points = np.asarray(selection['polygon'], np.int32)
        if len(points) < 3 or cv2.contourArea(points) < 10:
            raise ValueError('Đa giác cần ít nhất 3 đỉnh và có diện tích.')
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask
    if not selection.get('rect'):
        raise ValueError('Hãy kéo khoanh vùng quanh vật cần đo.')
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = selection['rect']
    x1, x2 = sorted((int(np.clip(x1, 0, w-1)), int(np.clip(x2, 0, w-1))))
    y1, y2 = sorted((int(np.clip(y1, 0, h-1)), int(np.clip(y2, 0, h-1))))
    if x2-x1 < 5 or y2-y1 < 5:
        raise ValueError('Vùng chọn quá nhỏ.')
    labels = np.full((h, w), cv2.GC_BGD, np.uint8)
    labels[y1:y2+1, x1:x2+1] = cv2.GC_PR_FGD
    for keep, points in selection.get('strokes', []):
        value = cv2.GC_FGD if keep else cv2.GC_BGD
        for p in points:
            cv2.circle(labels, tuple(map(int, p)), 4, value, -1)
        for a, b in zip(points, points[1:]):
            cv2.line(labels, tuple(map(int, a)), tuple(map(int, b)), value, 9)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(rgb, labels, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError('Không tách được vùng; hãy thêm nét giữ vật hoặc loại nền.')
    votes = [components[y,x] for keep, pts in selection.get('strokes', []) if keep
             for x,y in pts if 0 <= x < w and 0 <= y < h and components[y,x] > 0]
    chosen = int(np.bincount(votes).argmax()) if votes else 1+int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return np.where(components == chosen, 255, 0).astype(np.uint8)
