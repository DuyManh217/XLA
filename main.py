"""Optional command-line interface; normal use: python app.py."""
import argparse
from pipeline import Settings, process, measurements, export_bundle


def main():
    parser = argparse.ArgumentParser(description='Đo kích thước vật thể; bỏ --reference để đo pixel.')
    parser.add_argument('image', help='Đường dẫn ảnh')
    parser.add_argument('--output', default='results.zip')
    parser.add_argument('--method', choices=['Canny', 'Otsu light', 'Otsu dark'], default='Canny')
    parser.add_argument('--reference', type=int, help='ID vật chuẩn (từ trái sang phải)')
    parser.add_argument('--size', type=float, help='Kích thước cạnh chuẩn')
    parser.add_argument('--axis', choices=['long', 'short'], default='long')
    parser.add_argument('--unit', choices=['mm', 'cm', 'in'], default='mm')
    args = parser.parse_args()
    try:
        result = process(args.image, Settings(method=args.method))
        calibration = dict(reference_id=args.reference, reference_size=args.size, unit=args.unit, reference_axis=args.axis) if args.reference is not None else None
        rows = measurements(result, **(calibration or {}))
        export_bundle(args.output, result, rows, calibration)
        print(f'{len(rows)} vật thể; {result.seconds:.2f}s; lưu tại {args.output}')
        for row in rows:
            print(row)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Lỗi: {error}\n')


if __name__ == '__main__':
    main()
