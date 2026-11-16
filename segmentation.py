"""Phân vùng có hướng dẫn; tọa độ tính trên ảnh xử lý, không phải canvas."""
import cv2
import numpy as np


def grabcut_mask(rgb, selection):
    h, w = rgb.shape[:2]
    if selection and selection.get('polygon'):
        points = np.asarray(selection['polygon'], np.int32)
        if len(points) < 3 or cv2.contourArea(points) < 10:
            raise ValueError('Chọn ít nhất 3 đỉnh tạo thành vùng có diện tích.')
        # Nếu chỉ vẽ đa giác thì lấy nguyên vùng đa giác; nếu có tô giữ/loại thì chạy GrabCut bên trong đa giác.
        if not selection.get('strokes'):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [points], 255)
            return mask
        labels = np.full((h, w), cv2.GC_BGD, np.uint8)
        cv2.fillPoly(labels, [points], cv2.GC_PR_FGD)
    elif selection and selection.get('rect') is not None:
        x1, y1, x2, y2 = selection['rect']
        x1, x2 = sorted((max(0, min(w - 1, int(x1))), max(0, min(w - 1, int(x2)))))
        y1, y2 = sorted((max(0, min(h - 1, int(y1))), max(0, min(h - 1, int(y2)))))
        if x2 - x1 < 5 or y2 - y1 < 5:
            raise ValueError('Vùng chọn quá nhỏ.')
        labels = np.full((h, w), cv2.GC_BGD, np.uint8)
        labels[y1:y2+1, x1:x2+1] = cv2.GC_PR_FGD
    else:
        raise ValueError('Hãy kéo khoanh vùng hoặc vẽ đa giác quanh vật trước.')
    # Mỗi nét gồm nhãn giữ/loại và danh sách điểm nối liên tiếp.
    for keep, points in selection.get('strokes', []):
        value = cv2.GC_FGD if keep else cv2.GC_BGD
        for x, y in points:
            cv2.circle(labels, (int(x), int(y)), 4, value, -1)
        for a, b in zip(points, points[1:]):
            cv2.line(labels, tuple(map(int, a)), tuple(map(int, b)), value, 9)
    if not np.any(labels == cv2.GC_BGD):
        raise ValueError('Cần để lại nền ngoài vùng chọn hoặc tô nét loại nền.')
    cv2.grabCut(rgb, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Ưu tiên vùng chứa nét giữ vật; nếu chưa tô, đề xuất vùng lớn nhất.
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise ValueError('Chưa tách được vật. Thêm nét giữ vật/loại nền rồi thử lại.')
    votes = []
    for keep, points in selection.get('strokes', []):
        if keep:
            votes.extend(components[min(h-1, max(0, int(y))), min(w-1, max(0, int(x)))] for x, y in points)
    votes = [v for v in votes if v > 0]
    chosen = int(np.bincount(votes).argmax()) if votes else 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return np.where(components == chosen, 255, 0).astype(np.uint8)
