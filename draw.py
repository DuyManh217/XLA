import numpy as np


def draw_line(img, pt1, pt2, color, thickness=1):
    """
    Vẽ đường thẳng nối 2 điểm (x1, y1) và (x2, y2) bằng thuật toán Bresenham.
    """
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Hướng đi (1 hoặc -1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    while True:
        # Vẽ độ dày (tạo thành hình vuông nhỏ quanh tâm điểm ảnh)
        # Nếu thickness = 2, nó sẽ tô các pixel xung quanh (x, y)
        half_t = thickness // 2
        for i in range(-half_t, half_t + 1):
            for j in range(-half_t, half_t + 1):
                px, py = x1 + i, y1 + j
                # Kiểm tra biên ảnh
                if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                    img[py, px] = color

        # Đã vẽ đến điểm cuối thì dừng
        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def draw_polygon(img, points, color, thickness=1, is_closed=True):
    """
    Vẽ đa giác (hình chữ nhật, hình tam giác...) lên ảnh.
    :param img: Ảnh cần vẽ (mảng NumPy)
    :param points: Mảng các đỉnh (shape: N x 2)
    :param color: Màu (B, G, R)
    :param thickness: Độ dày nét vẽ
    :param is_closed: True nếu muốn nối điểm cuối với điểm đầu (khép kín)
    """
    num_pts = len(points)

    for i in range(num_pts - 1):
        draw_line(img, points[i], points[i + 1], color, thickness)

    # Nối điểm cuối với điểm đầu để khép kín hình
    if is_closed:
        draw_line(img, points[-1], points[0], color, thickness)


def draw_circle(img, center, radius, color, thickness=-1):
    """
    Vẽ hình tròn (dùng phương trình đường tròn (x-x0)^2 + (y-y0)^2 <= r^2)
    """
    cx, cy = int(center[0]), int(center[1])

    # Quét qua một hình vuông bao quanh hình tròn
    for i in range(cx - radius, cx + radius + 1):
        for j in range(cy - radius, cy + radius + 1):
            # Nếu nằm trong biên ảnh
            if 0 <= i < img.shape[1] and 0 <= j < img.shape[0]:
                if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                    # Nếu vẽ rỗng (thickness > 0), chỉ vẽ viền
                    if thickness > 0:
                        if (i - cx) ** 2 + (j - cy) ** 2 >= (radius - thickness) ** 2:
                            img[j, i] = color
                    else:
                        # Tô đặc
                        img[j, i] = color
