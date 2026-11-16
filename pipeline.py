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
from contours import sort_contours, contour_area
from io_utils import bgr_to_gray
from spatial_filters import gaussian_blur
from morphology import closing, fill_holes
from segmentation import grabcut_mask
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
    area_mode: str = 'contour'  # contour, filled, pixels

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
        if self.area_mode not in ('contour', 'filled', 'pixels'):
            raise ValueError('Cách tính diện tích không hợp lệ')


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
    blur = gaussian_blur(gray, settings.blur)
    if settings.method == 'GrabCut':
        binary = grabcut_mask(rgb, selection) if selection else np.zeros(gray.shape, np.uint8)
    elif settings.method == 'Canny':
        binary = cv2.Canny(blur, settings.low, settings.high)
    else:
        binary, _ = otsu_thresholding(blur)
        if settings.method == 'Otsu dark':
            binary = 255 - binary
    closed = binary if settings.method == 'GrabCut' else closing(binary, np.ones((settings.close, settings.close), dtype=np.uint8))
    mask = fill_holes(closed) if settings.area_mode == 'filled' else closed
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = sort_contours(contours)
    objects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if settings.area_mode == 'contour':
            area = contour_area(c)
        else:
            region = np.zeros((ch, cw), np.uint8)
            cv2.drawContours(region, [c - (x, y)], -1, 255, -1)
            area = np.count_nonzero((mask[y:y+ch, x:x+cw] > 0) & (region > 0))
        if area < settings.min_area:
            continue
        # Morphology erodes edge pixels: also reject contours within kernel radius.
        margin = settings.close // 2 + 1
        if settings.exclude_border and (x <= margin or y <= margin or x+cw >= size[0]-margin or y+ch >= size[1]-margin):
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
        objects.append(dict(id=len(objects)+1, long_px=float(long_px), short_px=float(short_px),
                            area_px2=float(contour_area(native) if settings.area_mode == 'contour' else area * sx * sy),
                            perimeter_px=float(cv2.arcLength(native, True)),
                            contour=c.reshape(-1, 2).tolist(), box=display_box.tolist()))
    stages = {'Ảnh gốc': rgb, 'Ảnh xám': gray, 'Lọc Gaussian': blur,
              'Biên / Nhị phân': binary, 'Đóng hình thái': closed, 'Mặt nạ đo': mask}
    overlay = rgb.copy()
    overlay[mask > 0] = (.55 * rgb[mask > 0] + .45 * np.array([30, 210, 130])).astype(np.uint8)
    stages['Vùng vật'] = overlay
    return Result(str(Path(path).resolve()), (w, h), settings, stages, objects,
                  time.perf_counter()-start, sx, sy)



def apply_grabcut(prev_result, settings, selection, keep_existing=True):
    """Sửa vùng đã chọn và giữ những vật cũ nằm ngoài vùng đó."""
    if settings.method != 'GrabCut' or not selection:
        raise ValueError('Hãy chọn GrabCut và khoanh vùng cần sửa.')
    if settings.max_side != prev_result.settings.max_side:
        raise ValueError('Kích thước xử lý đã đổi; hãy chọn lại vùng trên ảnh.')

    result = process(prev_result.source, settings, selection)
    if not result.objects:
        raise ValueError('Không tách được vật hợp lệ trong vùng chọn.')

    if keep_existing:
        def inside(obj):
            cx, cy = np.mean(obj['box'], axis=0)
            if selection.get('polygon'):
                polygon = np.asarray(selection['polygon'], np.float32)
                return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0
            x1, y1, x2, y2 = selection['rect']
            return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)

        kept = [dict(obj) for obj in prev_result.objects if not inside(obj)]
        next_id = max((obj['id'] for obj in kept), default=0) + 1
        for obj in result.objects:
            obj['id'] = next_id
            next_id += 1
        result.objects = kept + result.objects

    # Hiển thị cả vùng cũ lẫn vùng GrabCut mới, kể cả khi vật có lỗ thật.
    rgb = result.stages['Ảnh gốc']
    mask = result.stages['Mặt nạ đo'].copy()
    if keep_existing:
        old_mask = prev_result.stages['Mặt nạ đo']
        for obj in kept:
            region = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(region, [np.asarray(obj['contour'], np.int32)], -1, 255, -1)
            mask[(old_mask > 0) & (region > 0)] = 255
    overlay = rgb.copy()
    overlay[mask > 0] = (.55 * rgb[mask > 0] + .45 * np.array([30, 210, 130])).astype(np.uint8)
    result.stages['Vùng vật'] = overlay
    result.stages['Mặt nạ đo'] = mask
    return result


def measurements(result, reference_id=None, reference_size=None, unit='mm', reference_axis='long'):
    factor = 1.0
    if reference_id is not None:
        if unit not in ('mm', 'cm', 'in') or reference_axis not in ('long', 'short'):
            raise ValueError('Đơn vị hoặc cạnh chuẩn không hợp lệ')
        if reference_size is None or not np.isfinite(reference_size) or reference_size <= 0:
            raise ValueError('Kích thước vật chuẩn phải là số hữu hạn lớn hơn 0')
        ref = next((o for o in result.objects if o['id'] == reference_id), None)
        if ref is None:
            raise ValueError('Không tìm thấy vật chuẩn; hãy chọn lại sau khi xử lý ảnh')
        factor = ref[reference_axis+'_px'] / reference_size
    else:
        unit = 'px'
    return [dict(id=o['id'], is_reference=o['id'] == reference_id,
                 long=round(o['long_px']/factor, 4), short=round(o['short_px']/factor, 4),
                 unit=unit, area_px2=round(o['area_px2'], 2),
                 area=round(o['area_px2']/factor**2, 4), area_unit=f'{unit}²',
                 perimeter=round(o['perimeter_px']/factor, 4)) for o in result.objects]


def annotate(result, rows, selected_id=None):
    image = result.stages['Ảnh gốc'].copy()
    for obj, row in zip(result.objects, rows):
        cv2.polylines(image, [np.asarray(obj['contour'], np.int32)], True, (80, 180, 255), 1)
        box = np.rint(obj['box']).astype(np.int32)
        color = (251, 191, 36) if row['is_reference'] else (36, 220, 168)
        if obj['id'] == selected_id:
            color = (100, 180, 255)
        cv2.polylines(image, [box], True, color, 2, cv2.LINE_AA)
        x = max(2, min(image.shape[1]-220, int(box[:, 0].min())))
        y = max(22, int(box[:, 1].min())-9)
        label = f"#{row['id']}  {row['long']:.1f} x {row['short']:.1f} {row['unit']}"
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (12, 20, 30), 4, cv2.LINE_AA)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def export_bundle(path, result, rows, calibration=None):
    """Single ZIP with annotated PNG, stages, UTF-8 BOM CSV and provenance JSON."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        images = {'result': annotate(result, rows)}
        for i, stage in enumerate(result.stages.values()):
            images[f'stage_{i}'] = stage
        for name, array in images.items():
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format='PNG')
            archive.writestr(name+'.png', stream.getvalue())
        table = io.StringIO(newline='')
        writer = csv.DictWriter(
            table,
            fieldnames=[
                'id',
                'is_reference',
                'long',
                'short',
                'unit',
                'area_px2',
                'area',
                'area_unit',
                'perimeter',
            ]
        )
        writer.writeheader()
        writer.writerows(rows)
        archive.writestr('measurements.csv', table.getvalue().encode('utf-8-sig'))
        meta = dict(source_filename=Path(result.source).name, original_size=result.original_size,
                    settings=asdict(result.settings), processing_seconds=result.seconds,
                    calibration=calibration, measurements=rows,
                    stages=list(result.stages),
                    note='Estimated rotated bounding-box dimensions. Reference and objects must share a plane; no perspective correction.')
        archive.writestr('metadata.json', json.dumps(meta, ensure_ascii=False, indent=2))
