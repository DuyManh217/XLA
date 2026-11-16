"""Shared processing, calibration and export layer; no GUI dependencies."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import io
import json
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps
from contours import sort_contours, contour_area, contour_perimeter
from segmentation import grabcut_mask
from io_utils import bgr_to_gray
from spatial_filters import gaussian_blur
from morphology import closing
from thresholding import otsu_thresholding

MAX_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class Settings:
    method: str = 'Canny'
    blur: int = 7
    low: int = 50
    high: int = 100
    close: int = 3
    min_area: float = 100.0  # working-image pixels squared
    max_side: int = 1400
    exclude_border: bool = True

    def validate(self):
        if self.method not in ('Canny', 'Otsu light', 'Otsu dark', 'GrabCut'):
            raise ValueError('Phương pháp không hợp lệ')
        if self.blur not in (1, 3, 5, 7, 9, 11, 15) or self.close not in (1, 3, 5, 7, 9):
            raise ValueError('Kích thước kernel không hợp lệ')
        if not 0 <= self.low < self.high <= 255:
            raise ValueError('Canny cần 0 ≤ ngưỡng thấp < ngưỡng cao ≤ 255')
        if not np.isfinite(self.min_area) or self.min_area < 0:
            raise ValueError('Diện tích tối thiểu phải hữu hạn và không âm')
        if not 200 <= self.max_side <= 2400:
            raise ValueError('Cạnh xử lý phải trong khoảng 200–2400 px')


@dataclass
class Result:
    source: str
    original_size: tuple
    settings: Settings
    stages: dict
    objects: list
    seconds: float
    scale_x: float
    scale_y: float
    selection: dict = None


def load_image(path):
    """Pillow handles Unicode paths, EXIF orientation, alpha and grayscale."""
    with Image.open(path) as raw:
        if raw.width * raw.height > MAX_IMAGE_PIXELS:
            raise ValueError('Ảnh quá lớn (tối đa 40 triệu pixel). Hãy giảm kích thước ảnh.')
        image = ImageOps.exif_transpose(raw)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(background, rgba)
        return np.asarray(image.convert('RGB')).copy()


def _extract_objects(mask, settings, size, sx, sy, start_id=1):
    """Tách danh sách đối tượng và quy đổi kích thước về độ phân giải ảnh gốc."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = sort_contours(contours)
    objects = []
    margin = settings.close // 2 + 1
    for c in contours:
        area = contour_area(c)
        if area < settings.min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        # Morphology làm mòn biên nên bỏ qua các contour sát viền ảnh nếu bật tùy chọn.
        if settings.exclude_border and (x <= margin or y <= margin or x + cw >= size[0] - margin or y + ch >= size[1] - margin):
            continue
        native = c.astype(np.float32).copy()
        native[:, 0, 0] *= sx
        native[:, 0, 1] *= sy
        rect = cv2.minAreaRect(native)
        long_px, short_px = sorted(rect[1], reverse=True)
        if short_px < 1e-6:
            continue
        native_box = cv2.boxPoints(rect)
        display_box = native_box / np.array([sx, sy])
        objects.append(dict(id=start_id + len(objects), long_px=float(long_px), short_px=float(short_px),
                            area_px2=float(contour_area(native)),
                            perimeter_px=float(contour_perimeter(native)),
                            contour=c.reshape(-1, 2).tolist(), box=display_box.tolist()))
    return objects


def _point_in_selection(px, py, selection):
    """Kiểm tra điểm (px, py) có nằm trong vùng khoanh hoặc đa giác đang chọn không."""
    if selection.get('polygon'):
        poly = np.asarray(selection['polygon'], dtype=np.float32)
        return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0
    if selection.get('rect'):
        x1, y1, x2, y2 = selection['rect']
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return x1 <= px <= x2 and y1 <= py <= y2
    return False


def _make_overlay(rgb, mask):
    """Tô lớp phủ màu xanh lá lên vùng vật thể để kiểm tra trực quan."""
    overlay = rgb.copy()
    overlay[mask > 0] = (0.55 * rgb[mask > 0] + 0.45 * np.array([30, 220, 130])).astype(np.uint8)
    return overlay


def process(path, settings=None, selection=None):
    start = time.perf_counter()
    settings = settings or Settings()
    settings.validate()
    original = load_image(path)
    h, w = original.shape[:2]
    ratio = min(1.0, settings.max_side / max(h, w))
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    rgb = np.asarray(Image.fromarray(original).resize(size, Image.Resampling.LANCZOS)) if ratio < 1 else original
    sx, sy = w / rgb.shape[1], h / rgb.shape[0]
    gray = bgr_to_gray(rgb[:, :, ::-1])
    blur = cv2.GaussianBlur(gray, (settings.blur, settings.blur), 0)
    if settings.method == 'GrabCut':
        binary = grabcut_mask(rgb, selection) if selection else np.zeros(gray.shape, np.uint8)
    elif settings.method == 'Canny':
        binary = cv2.Canny(blur, settings.low, settings.high)
    else:
        binary, _ = otsu_thresholding(blur)
        if settings.method == 'Otsu dark':
            binary = 255 - binary
    mask = binary if settings.method == 'GrabCut' else cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((settings.close, settings.close), np.uint8))
    objects = _extract_objects(mask, settings, size, sx, sy)
    stages = {'Ảnh gốc': rgb, 'Ảnh xám': gray, 'Lọc Gaussian': blur,
              'Biên / Nhị phân': binary, 'Đóng hình thái': mask,
              'Vùng vật': _make_overlay(rgb, mask)}
    return Result(str(Path(path).resolve()), (w, h), settings, stages, objects,
                  time.perf_counter()-start, sx, sy, selection)


def apply_grabcut(prev_result, settings, selection, keep_existing=True):
    """Bổ sung hoặc thay thế vật trong vùng khoanh bằng GrabCut, giữ nguyên các vật cũ bên ngoài."""
    start = time.perf_counter()
    settings.validate()
    rgb = prev_result.stages['Ảnh gốc']
    size = (rgb.shape[1], rgb.shape[0])
    mask = grabcut_mask(rgb, selection)
    if keep_existing and prev_result.objects:
        kept = []
        for obj in prev_result.objects:
            cx, cy = np.mean(obj['box'], axis=0)
            if not _point_in_selection(cx, cy, selection):
                kept.append(dict(obj))
        next_id = max((o['id'] for o in kept), default=0) + 1
        added = _extract_objects(mask, settings, size, prev_result.scale_x, prev_result.scale_y, start_id=next_id)
        if not added:
            raise ValueError('Không tách được vật hợp lệ trong vùng chọn (quá nhỏ hoặc sát mép ảnh).')
        objects = kept + added
    else:
        objects = _extract_objects(mask, settings, size, prev_result.scale_x, prev_result.scale_y)
    # Dựng lại mặt nạ tổng hợp để bước 'Vùng vật' hiển thị cả vật cũ lẫn vật mới.
    all_mask = np.zeros(rgb.shape[:2], np.uint8)
    for obj in objects:
        cv2.drawContours(all_mask, [np.array(obj['contour'], np.int32)], -1, 255, -1)
    stages = dict(prev_result.stages)
    stages['Biên / Nhị phân'] = mask
    stages['Đóng hình thái'] = mask
    stages['Vùng vật'] = _make_overlay(rgb, all_mask)
    return Result(prev_result.source, prev_result.original_size, settings, stages, objects,
                  time.perf_counter() - start, prev_result.scale_x, prev_result.scale_y, selection)


def measurements(result, reference_id=None, reference_size=None, unit='mm', reference_axis='long', reference_points=None):
    factor = 1.0
    if reference_id is not None or reference_points is not None:
        if unit not in ('mm', 'cm', 'in') or reference_axis not in ('long', 'short'):
            raise ValueError('Đơn vị hoặc cạnh chuẩn không hợp lệ')
        if reference_size is None or not np.isfinite(reference_size) or reference_size <= 0:
            raise ValueError('Kích thước vật chuẩn phải là số hữu hạn lớn hơn 0')
        if reference_points is not None:
            points = np.asarray(reference_points, dtype=float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError('Cần hai điểm chuẩn hợp lệ.')
            distance = np.linalg.norm((points[1] - points[0]) * [result.scale_x, result.scale_y])
        else:
            ref = next((o for o in result.objects if o['id'] == reference_id), None)
            if ref is None:
                raise ValueError('Không tìm thấy vật chuẩn; hãy chọn lại sau khi xử lý ảnh')
            distance = ref[reference_axis+'_px']
        if distance <= 0:
            raise ValueError('Hai điểm chuẩn phải khác nhau.')
        factor = distance / reference_size
    else:
        unit = 'px'
    return [dict(id=o['id'], is_reference=o['id'] == reference_id,
                 long=round(o['long_px']/factor, 4), short=round(o['short_px']/factor, 4),
                 unit=unit, area=round(o['area_px2']/factor**2, 4),
                 perimeter=round(o['perimeter_px']/factor, 4),
                 area_px2=round(o['area_px2'], 2)) for o in result.objects]


def annotate(result, rows, selected_id=None, calibration=None):
    image = result.stages['Ảnh gốc'].copy()
    for obj, row in zip(result.objects, rows):
        box = np.rint(obj['box']).astype(np.int32)
        color = (251, 191, 36) if row['is_reference'] else (36, 220, 168)
        if obj['id'] == selected_id:
            color = (100, 180, 255)
        cv2.polylines(image, [np.array(obj['contour'], np.int32)], True, (80, 180, 255), 1)
        cv2.polylines(image, [box], True, color, 2, cv2.LINE_AA)
        x = max(2, min(image.shape[1]-220, int(box[:, 0].min())))
        y = max(22, int(box[:, 1].min())-9)
        label = f"#{row['id']}  {row['long']:.1f} x {row['short']:.1f} {row['unit']}"
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (12, 20, 30), 4, cv2.LINE_AA)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    if calibration and calibration.get('reference_points'):
        a, b = np.rint(calibration['reference_points']).astype(int)
        cv2.line(image, tuple(a), tuple(b), (40, 160, 255), 2)
        cv2.putText(image, f"REF {calibration['reference_size']:g} {calibration['unit']}",
                    tuple(a), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 160, 255), 2)
    return image


def export_bundle(path, result, rows, calibration=None):
    """Single ZIP with annotated PNG, stages, UTF-8 BOM CSV and provenance JSON."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        images = {'result': annotate(result, rows, calibration=calibration)}
        for i, stage in enumerate(result.stages.values()):
            images[f'stage_{i}'] = stage
        for name, array in images.items():
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format='PNG')
            archive.writestr(name+'.png', stream.getvalue())
        table = io.StringIO(newline='')
        writer = csv.DictWriter(table, fieldnames=['id', 'is_reference', 'long', 'short', 'unit', 'area', 'perimeter', 'area_px2'])
        writer.writeheader()
        writer.writerows(rows)
        archive.writestr('measurements.csv', table.getvalue().encode('utf-8-sig'))
        meta = dict(source_filename=Path(result.source).name, original_size=result.original_size,
                    selection=result.selection,
                    settings=asdict(result.settings), processing_seconds=result.seconds,
                    calibration=calibration, measurements=rows,
                    stages=list(result.stages),
                    note='Estimated rotated bounding-box dimensions. Reference and objects must share a plane; no perspective correction.')
        archive.writestr('metadata.json', json.dumps(meta, ensure_ascii=False, indent=2))
