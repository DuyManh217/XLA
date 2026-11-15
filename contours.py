import numpy as np


def find_contours(binary_img):
    """
    Tìm các đường bao (contours) từ ảnh nhị phân (From scratch).
    Sử dụng thuật toán Loang (Flood Fill) để tìm các khối liên thông.
    """
    h, w = binary_img.shape
    visited = np.zeros((h, w), dtype=bool)
    contours = []

    # 8 hướng lân cận (bao gồm cả chéo)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for y in range(h):
        for x in range(w):
            # Nếu gặp điểm ảnh trắng (255) và chưa được thăm
            if binary_img[y, x] == 255 and not visited[y, x]:

                # 1. Dùng BFS (Hàng đợi) để loang tìm toàn bộ các pixel của vật thể
                queue = [(x, y)]
                visited[y, x] = True
                component_pixels = []

                while queue:
                    cx, cy = queue.pop(0)
                    component_pixels.append((cx, cy))

                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if binary_img[ny, nx] == 255 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((nx, ny))

                # 2. Lọc nhiễu: Chỉ lấy các vật thể có diện tích > 50 pixel
                if len(component_pixels) > 50:

                    # 3. Tìm đường viền (Boundary) của vật thể này
                    # Đường viền là những điểm thuộc vật thể nhưng có ít nhất 1 lân cận là nền (0)
                    boundary = []
                    for px, py in component_pixels:
                        is_boundary = False
                        for dx, dy in [
                            (-1, 0),
                            (1, 0),
                            (0, -1),
                            (0, 1),
                        ]:  # Chỉ cần xét 4 hướng
                            nx, ny = px + dx, py + dy
                            # Nếu nằm sát biên ảnh hoặc lân cận là nền
                            if (
                                nx < 0
                                or nx >= w
                                or ny < 0
                                or ny >= h
                                or binary_img[ny, nx] == 0
                            ):
                                is_boundary = True
                                break
                        if is_boundary:
                            boundary.append((px, py))

                    # 4. Chuyển đổi sang định dạng giống OpenCV (N, 1, 2)
                    if len(boundary) > 0:
                        contour_np = np.array(boundary, dtype=np.int32).reshape(
                            -1, 1, 2
                        )
                        contours.append(contour_np)

    return contours


def sort_contours(cnts, method="left-to-right"):
    """
    Sắp xếp danh sách các contours theo thứ tự hình học.
    """
    boundingBoxes = []

    for c in cnts:
        x_coords = c[:, 0, 0]
        y_coords = c[:, 0, 1]

        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        w = x_max - x_min
        h = y_max - y_min

        boundingBoxes.append((x_min, y_min, w, h))

    boundingBoxes = np.array(boundingBoxes)

    if method == "left-to-right":
        idxs = np.argsort(boundingBoxes[:, 0])
    elif method == "right-to-left":
        idxs = np.argsort(boundingBoxes[:, 0])[::-1]
    elif method == "top-to-bottom":
        idxs = np.argsort(boundingBoxes[:, 1])
    elif method == "bottom-to-top":
        idxs = np.argsort(boundingBoxes[:, 1])[::-1]
    else:
        idxs = np.argsort(boundingBoxes[:, 0])

    sorted_cnts = [cnts[i] for i in idxs]
    sorted_boxes = boundingBoxes[idxs]

    return sorted_cnts, sorted_boxes


# ==========================================
# CÁC HÀM THAY THẾ OPENCV (TỪ SCRATCH)
# ==========================================


def contour_area(c):
    """
    Tính diện tích contour bằng công thức Shoelace (thay cho cv2.contourArea).
    """
    pts = c.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def box_points(rect):
    """
    Trả về 4 tọa độ đỉnh của hình chữ nhật xoay (thay cho cv2.boxPoints).
    rect là tuple: ((center_x, center_y), (width, height), angle)
    """
    (cx, cy), (w, h), angle = rect
    # Chuyển đổi góc sang radian
    theta = np.deg2rad(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Tọa độ 4 góc trước khi xoay so với tâm
    dx = w / 2.0
    dy = h / 2.0

    # 4 điểm: (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])

    # Ma trận xoay
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Xoay và tịnh tiến về tâm thực tế
    rotated_corners = np.dot(corners, rotation_matrix.T) + [cx, cy]
    return rotated_corners.astype(np.int32)


def min_area_rect(c):
    """
    Tìm hình chữ nhật xoay nhỏ nhất bao quanh contour (thay cho cv2.minAreaRect).
    Sử dụng phương pháp xoay trục tọa độ (Rotating Calipers đơn giản hóa).
    """
    pts = c.reshape(-1, 2).astype(np.float32)

    # Nếu ít hơn 3 điểm, trả về hộp thẳng đứng
    if len(pts) < 3:
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        return (
            ((x_min + x_max) / 2, (y_min + y_max) / 2),
            (x_max - x_min, y_max - y_min),
            0.0,
        )

    # Thuật toán: Tìm góc xoay sao cho diện tích hộp bao là nhỏ nhất
    # Quét các góc từ 0 đến 90 độ (bước 1 độ để chạy nhanh)
    min_area = float("inf")
    best_rect = None

    # Để tối ưu, ta chỉ cần quét góc dựa trên các cạnh của đa giác lồi
    # Nhưng để code đơn giản, ta quét góc từ 0 đến 90 độ
    for angle in range(0, 90, 2):
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Xoay tất cả các điểm
        rotation_matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        rotated_pts = np.dot(pts, rotation_matrix.T)

        # Tìm hộp bao thẳng đứng của các điểm đã xoay
        x_min, y_min = np.min(rotated_pts, axis=0)
        x_max, y_max = np.max(rotated_pts, axis=0)

        w = x_max - x_min
        h = y_max - y_min
        area = w * h

        if area < min_area:
            min_area = area

            # Tọa độ tâm trong hệ trục đã xoay
            cx_rot = (x_min + x_max) / 2
            cy_rot = (y_min + y_max) / 2

            # Xoay tâm ngược lại về hệ trục gốc
            cx = cx_rot * cos_t - cy_rot * sin_t
            cy = cx_rot * sin_t + cy_rot * cos_t

            best_rect = ((cx, cy), (w, h), angle)

    return best_rect
