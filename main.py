import numpy as np
import imutils
import cv2
from contours import (
    find_contours,
    sort_contours,
    contour_area,
    min_area_rect,
    box_points,
)
from draw import draw_circle, draw_line, draw_polygon
from edge_detection import canny_edge_detection
from io_utils import bgr_to_gray, euclidean_dist, midpoint, order_points
from spatial_filters import gaussian_blur
from morphology import closing


def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()


INPUT_PATH = "images/example_02.png"
REFERENCE_WIDTH = 0.955
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PINK = (255, 0, 255)
WHITE = (255, 255, 255)

image = cv2.imread(INPUT_PATH)
if image is None:
    raise FileNotFoundError("could not read image: {}".format(INPUT_PATH))
gray = bgr_to_gray(image)
gray = gaussian_blur(gray, ksize=7, sigma=0)


edged = canny_edge_detection(gray, 50, 100)
show_image("Edged", edged, False)
kernel = np.ones((3, 3), dtype=np.uint8)
edged = closing(edged, kernel)
show_image("erode and dilate", edged, True)


cnts = find_contours(edged)
print("Total number of contours are: ", len(cnts))

cnts, _ = sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
count = 0
for c in cnts:
    # 1. Lọc nhiễu
    if contour_area(c) < 100:
        continue
    count += 1

    # 2. Tìm hình chữ nhật xoay và sắp xếp 4 góc
    # orig = image.copy()
    orig = image
    rect = min_area_rect(c)
    box = box_points(rect)
    box = np.array(box, dtype="int")
    box = order_points(box)  # Hàm tự viết

    # 3. Unpack 4 góc
    tl, tr, br, bl = box

    # 4. TÍNH TOÁN TRƯỚC (Midpoints và Khoảng cách)
    tltrX, tltrY = midpoint(tl, tr)
    blbrX, blbrY = midpoint(bl, br)
    tlblX, tlblY = midpoint(tl, bl)
    trbrX, trbrY = midpoint(tr, br)

    dA = euclidean_dist((tltrX, tltrY), (blbrX, blbrY))
    dB = euclidean_dist((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = max(dA, dB) / REFERENCE_WIDTH

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # 5. BẮT ĐẦU VẼ (Sau khi đã có đủ biến)

    # 5.1 Vẽ khung chữ nhật xanh lá (Thay cv2.drawContours)
    draw_polygon(orig, box, color=(0, 255, 0), thickness=2)

    # 5.2 Vẽ 4 điểm góc màu đỏ (Thay cv2.circle)
    for x, y in box:
        draw_circle(orig, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # 5.3 Vẽ 4 điểm trung điểm màu xanh dương (Thay cv2.circle)
    draw_circle(orig, (int(tltrX), int(tltrY)), radius=5, color=RED, thickness=-1)
    draw_circle(orig, (int(blbrX), int(blbrY)), radius=5, color=RED, thickness=-1)
    draw_circle(orig, (int(tlblX), int(tlblY)), radius=5, color=RED, thickness=-1)
    draw_circle(orig, (int(trbrX), int(trbrY)), radius=5, color=RED, thickness=-1)

    # 5.4 Vẽ 2 đường chéo nối trung điểm màu hồng (Thay cv2.line)
    draw_line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), PINK, 2)
    draw_line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), PINK, 2)

    # 5.5 Viết kích thước lên ảnh (Dùng cv2.putText vì viết chữ from scratch rất phức tạp)
    cv2.putText(
        orig,
        "{:.1f}in".format(dimA),
        (int(tltrX - 15), int(tltrY - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
    )
    cv2.putText(
        orig,
        "{:.1f}in".format(dimB),
        (int(trbrX + 10), int(trbrY)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
    )

    # Hiển thị
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
