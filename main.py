import os
import csv
import argparse
import numpy as np
import cv2
from contours import (
    find_contours,
    sort_contours,
    contour_area,
    contour_perimeter,
    min_area_rect,
    box_points,
)
from draw import draw_circle, draw_line, draw_polygon
from edge_detection import canny_edge_detection
from io_utils import bgr_to_gray, euclidean_dist, midpoint, order_points
from spatial_filters import gaussian_blur
from morphology import closing, fill_holes
from measurement_utils import ROOT, setup_console, load_bgr, save_bgr, prepare_measurement


def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()


def main():
    setup_console()
    # 1. Nhận tham số từ dòng lệnh (argparse)
    parser = argparse.ArgumentParser(description="Đo kích thước đối tượng từ ảnh - BTL XLA")
    parser.add_argument("-i", "--image", default="images/example_02.png", help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("-w", "--width", type=float, default=0.955, help="Kích thước thực của vật chuẩn ngoài cùng bên trái")
    parser.add_argument("-u", "--unit", default="in", choices=["in", "mm", "cm"], help="Đơn vị đo (in, mm, cm)")
    parser.add_argument("--no-show", action="store_true", help="Không hiện cửa sổ GUI từng bước (chạy ngầm)")
    parser.add_argument("--reference", type=int, default=1, help="STT vật chuẩn từ trái sang phải, bắt đầu từ 1")
    parser.add_argument("--output", default=str(ROOT / "data/output"), help="Thư mục lưu kết quả")
    args = parser.parse_args()

    input_path = args.image
    reference_width = args.width
    unit = args.unit
    if not np.isfinite(reference_width) or reference_width <= 0:
        parser.error("--width phải là số hữu hạn lớn hơn 0")
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        input_path = str(ROOT / input_path)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PINK = (255, 0, 255)
    WHITE = (255, 255, 255)

    # 2. Đọc ảnh và chuyển mức xám
    image = load_bgr(input_path)
    if image is None:
        raise FileNotFoundError("could not read image: {}".format(input_path))
    gray = bgr_to_gray(image)
    gray = gaussian_blur(gray, ksize=7, sigma=0)

    # 3. Phát hiện biên và làm liền biên bằng hình thái học
    edged = canny_edge_detection(gray, 50, 100)
    if not args.no_show:
        show_image("Edged", edged, False)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edged = closing(edged, kernel)
    edged = fill_holes(edged)
    if not args.no_show:
        show_image("erode and dilate", edged, True)

    # 4. Tìm và sắp xếp contours từ trái sang phải
    cnts = find_contours(edged)
    print("Total number of contours are: ", len(cnts))

    cnts, pixelsPerMetric = prepare_measurement(cnts, reference_width, args.reference)

    # Khởi tạo danh sách lưu kết quả để xuất CSV
    results_list = []
    orig = image.copy()
    count = 0

    # 5. Lặp qua từng contour để đo đạc
    for c in cnts:
        # Lọc nhiễu: Bỏ qua contour quá nhỏ
        if contour_area(c) < 100:
            continue
        count += 1

        # Tìm hình chữ nhật xoay và sắp xếp 4 góc
        rect = min_area_rect(c)
        box = box_points(rect)
        box = order_points(box)

        # Unpack 4 góc
        tl, tr, br, bl = box

        # Tính trung điểm và khoảng cách
        tltrX, tltrY = midpoint(tl, tr)
        blbrX, blbrY = midpoint(bl, br)
        tlblX, tlblY = midpoint(tl, bl)
        trbrX, trbrY = midpoint(tr, br)

        dA = euclidean_dist((tltrX, tltrY), (blbrX, blbrY))
        dB = euclidean_dist((tlblX, tlblY), (trbrX, trbrY))

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Tính diện tích và chu vi thực tế
        c_area_px = contour_area(c)
        c_perim_px = contour_perimeter(c)
        area_real = c_area_px / (pixelsPerMetric ** 2)
        perim_real = c_perim_px / pixelsPerMetric

        # Lưu thông tin đối tượng
        role = "Vật chuẩn (Ref)" if count == args.reference else f"Vật thể #{count}"
        results_list.append({
            "STT": count,
            "Vai trò": role,
            f"Chiều rộng ({unit})": round(float(min(dimA, dimB)), 2),
            f"Chiều dài ({unit})": round(float(max(dimA, dimB)), 2),
            f"Diện tích ({unit}2)": round(float(area_real), 2),
            f"Chu vi ({unit})": round(float(perim_real), 2),
        })

        # Vẽ khung chữ nhật và các điểm đo
        draw_polygon(orig, box, color=(0, 255, 0), thickness=2)
        for x, y in box:
            draw_circle(orig, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        draw_circle(orig, (int(tltrX), int(tltrY)), radius=5, color=RED, thickness=-1)
        draw_circle(orig, (int(blbrX), int(blbrY)), radius=5, color=RED, thickness=-1)
        draw_circle(orig, (int(tlblX), int(tlblY)), radius=5, color=RED, thickness=-1)
        draw_circle(orig, (int(trbrX), int(trbrY)), radius=5, color=RED, thickness=-1)

        draw_line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), PINK, 2)
        draw_line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), PINK, 2)

        # Viết kích thước lên ảnh
        cv2.putText(orig, f"#{count}", tuple(box[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(
            orig,
            f"{dimA:.1f}{unit}",
            (int(tltrX - 15), int(tltrY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2,
        )
        cv2.putText(
            orig,
            f"{dimB:.1f}{unit}",
            (int(trbrX + 10), int(trbrY)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2,
        )

        if not args.no_show:
            cv2.imshow("Image", orig)
            cv2.waitKey(0)

    # 6. In bảng tổng kết và xuất file báo cáo
    print("\n" + "=" * 75)
    print(f"KẾT QUẢ ĐO LƯỜNG (PixelsPerMetric: {pixelsPerMetric:.2f} pixels/{unit})")
    print("=" * 75)
    header = f"{'STT':<4} | {'Vai trò':<16} | {'Rộng':<10} | {'Dài':<10} | {'Diện tích':<14} | {'Chu vi':<10}"
    print(header)
    print("-" * len(header))
    for r in results_list:
        w_val = f"{float(r[f'Chiều rộng ({unit})']):.2f} {unit}"
        h_val = f"{float(r[f'Chiều dài ({unit})']):.2f} {unit}"
        a_val = f"{float(r[f'Diện tích ({unit}2)']):.2f} {unit}2"
        p_val = f"{float(r[f'Chu vi ({unit})']):.2f} {unit}"
        print(f"{r['STT']:<4} | {r['Vai trò']:<16} | {w_val:<10} | {h_val:<10} | {a_val:<14} | {p_val:<10}")
    print("=" * 75)

    # Tự động lưu ảnh và file CSV vào data/output/
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    out_img_path = os.path.join(output_dir, "result.png")
    save_bgr(out_img_path, orig)
    print(f"[+] Đã lưu ảnh kết quả tại: {out_img_path}")

    out_csv_path = os.path.join(output_dir, "report.csv")
    if results_list:
        keys = results_list[0].keys()
        with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results_list)
        print(f"[+] Đã xuất file báo cáo tại: {out_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, cv2.error) as error:
        print(f"Lỗi: {error}")
        raise SystemExit(1)
