"""CLI dùng cùng pipeline với giao diện; thuật toán học tập giữ trong các module."""
import argparse
import csv
from pathlib import Path
from PIL import Image
from pipeline import Settings, process, measurements, annotate, export_bundle
from measurement_utils import ROOT, setup_console


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Đo vật từ ảnh; chưa có chuẩn thì chỉ đo pixel.")
    parser.add_argument("-i", "--image", default=str(ROOT / "images/example_02.png"))
    parser.add_argument("-w", "--width", type=float, help="Kích thước thật của cạnh chuẩn")
    parser.add_argument("--reference", type=int, default=1)
    parser.add_argument("--axis", choices=["long", "short"], default="long")
    parser.add_argument("-u", "--unit", choices=["mm", "cm", "in"], default="mm")
    parser.add_argument("--method", choices=["Canny", "Otsu light", "Otsu dark"], default="Canny")
    parser.add_argument("--output", default=str(ROOT / "data/output"))
    parser.add_argument("--no-show", action="store_true", help="Tương thích lệnh cũ; CLI luôn không mở cửa sổ.")
    args = parser.parse_args()
    try:
        result = process(args.image, Settings(method=args.method))
        if not result.objects:
            raise ValueError("Không tìm được vùng đo; thử giao diện để chọn và sửa vùng.")
        calibration = None
        if args.width is not None:
            calibration = dict(reference_id=args.reference, reference_size=args.width,
                               unit=args.unit, reference_axis=args.axis)
        rows = measurements(result, **(calibration or {}))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotate(result, rows)).save(output / "result.png")
        with (output / "report.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        export_bundle(output / "results.zip", result, rows, calibration)
        for row in rows:
            print(row)
        print("Đã xuất:", output)
        print("CLI: vùng đo chưa được xác nhận thủ công; cần xem ảnh kết quả.")
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
