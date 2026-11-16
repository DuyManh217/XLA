"""Đối chứng hình học trên cùng mask và sai số với số đo tham khảo."""
import json
import time
import cv2
import numpy as np
from contours import find_contours, contour_area, contour_perimeter, min_area_rect, sort_contours
from edge_detection import canny_edge_detection
from io_utils import bgr_to_gray
from morphology import closing, fill_holes
from spatial_filters import gaussian_blur
from measurement_utils import ROOT, setup_console, load_bgr


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def measure(contours, ref_width, library=False):
    area = cv2.contourArea if library else contour_area
    perimeter = cv2.arcLength if library else contour_perimeter
    rectangle = cv2.minAreaRect if library else min_area_rect
    contours, _ = sort_contours([c for c in contours if area(c) >= 100])
    if not contours:
        return [], []
    ppm = max(rectangle(contours[0])[1]) / ref_width
    rows = []
    for c in contours:
        w, h = sorted(rectangle(c)[1])
        rows.append(np.array([w / ppm, h / ppm, area(c) / ppm ** 2, perimeter(c, True) / ppm]))
    return contours, rows


def compare_measurements(path, truth):
    image = load_bgr(path)
    gray = bgr_to_gray(image)
    start = time.perf_counter()
    blur = gaussian_blur(gray)
    blur_time = time.perf_counter() - start
    blur_cv = cv2.GaussianBlur(gray, (7, 7), 0)
    print(f"\n{path.name}: Gaussian {blur_time:.3f}s, PSNR {compute_psnr(blur, blur_cv):.2f} dB")
    start = time.perf_counter()
    edges = canny_edge_detection(blur)
    elapsed = time.perf_counter() - start
    edges_cv = cv2.Canny(blur, 50, 100)
    union = np.count_nonzero((edges > 0) | (edges_cv > 0))
    intersection = np.count_nonzero((edges > 0) & (edges_cv > 0))
    print(f"Canny scratch: {elapsed:.3f}s; IoU biên: {intersection / union if union else 1:.3f}")
    mask = fill_holes(closing(edges, np.ones((3, 3), dtype=np.uint8)))
    ref = truth["reference_object"]["width_in"]
    scratch, rows = measure(find_contours(mask), ref)
    cv_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    standard, expected = measure(cv_contours, ref, library=True)
    print(f"Số vật scratch/OpenCV trên cùng mask: {len(rows)}/{len(expected)}")
    print("STT | lệch rộng % | lệch dài % | lệch diện tích % | lệch chu vi %")
    errors = []
    # Cùng mask phải có cùng vị trí hộp bao; không ghép mù theo STT.
    unused = list(range(len(standard)))
    for i, c in enumerate(scratch):
        match = next((j for j in unused if cv2.boundingRect(c) == cv2.boundingRect(standard[j])), None)
        if match is None:
            print(f"{i + 1}: KHÔNG GHÉP ĐƯỢC")
            continue
        unused.remove(match)
        err = np.abs(rows[i] - expected[match]) / np.maximum(expected[match], 1e-12) * 100
        errors.append(err)
        print(i + 1, " | ", " | ".join(f"{v:.3f}" for v in err))
    if errors:
        print("Sai lệch trung bình W/H/A/P (%):", np.mean(errors, axis=0).round(3))
    print(f"Chưa ghép: scratch={len(rows) - len(errors)}, OpenCV={len(unused)}")
    # Số đo khai báo chưa thay thế thí nghiệm đo bằng thước.
    targets = truth.get("targets_ground_truth", [])
    for item in targets:
        index = item["id"] - 1
        if index >= len(rows):
            print("Thiếu đối tượng tham khảo:", item["name"])
            continue
        target = np.sort([item["width_in"], item["height_in"]])
        err = np.abs(rows[index][:2] - target) / target * 100
        print(f"Tham khảo {item['name']}: W/H={rows[index][:2].round(3)}, sai số %={err.round(3)}")
    if not targets:
        print("Chưa có kích thước vật đích đo thực tế; không kết luận độ chính xác vật lý.")


if __name__ == "__main__":
    setup_console()
    truth = json.loads((ROOT / "data/input/ground_truth.json").read_text(encoding="utf-8"))
    for name, data in truth.items():
        compare_measurements(ROOT / "images" / name, data)
    print("\nOpenCV chỉ là đối chứng thuật toán. Không quy đổi PSNR thành % chính xác.")
