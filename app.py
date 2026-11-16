"""Run: python app.py. Desktop UI; calculations run on a background thread."""
from pathlib import Path
import queue
import threading
import copy
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from pipeline import Settings, process, apply_grabcut, measurements, annotate, export_bundle

ROOT = Path(__file__).resolve().parent


class ImagePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.array = None
        self.photo = None
        self.pending = None
        self.canvas = tk.Canvas(self, bg='#101a2d', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.schedule)

    def schedule(self, event=None):
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = self.after(60, self.render)

    def show(self, array):
        self.array = array
        self.render()

    def render(self):
        self.pending = None
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.array is None:
            self.canvas.create_text(w/2, h/2, text='Chọn ảnh để bắt đầu', fill='#aab9cf', font=('Segoe UI', 17))
            return
        image = Image.fromarray(self.array)
        image.thumbnail((max(1, w-24), max(1, h-24)), Image.Resampling.LANCZOS)
        self.origin = ((w-image.width)/2, (h-image.height)/2)
        self.ratio = (image.width/self.array.shape[1], image.height/self.array.shape[0])
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(w/2, h/2, image=self.photo)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('XLA • Đo kích thước vật thể')
        self.geometry('1260x840')
        self.minsize(1000, 740)
        self.configure(bg='#f3f6fb')
        self.path = None
        self.result = None
        self.rows = []
        self.calibration = None
        self.busy = False
        self.dirty = False
        self.messages = queue.Queue()
        self.status = tk.StringVar(value='Sẵn sàng • Chọn ảnh của bạn hoặc mở một ảnh mẫu.')
        self.summary = tk.StringVar(value='Chưa có ảnh')
        self.stage = tk.StringVar(value='Kết quả')
        self.method = tk.StringVar(value='Canny')
        self.tool = tk.StringVar(value='Khoanh vùng')
        self.keep_existing = tk.BooleanVar(value=True)
        self.confirmed = tk.BooleanVar(value=False)
        self.selection = {}
        self.reference_points = []
        self.drag_start = None
        self.blur = tk.StringVar(value='7')
        self.low = tk.StringVar(value='50')
        self.high = tk.StringVar(value='100')
        self.close_kernel = tk.StringVar(value='3')
        self.min_area = tk.StringVar(value='100')
        self.max_side = tk.StringVar(value='1000')
        self.exclude_border = tk.BooleanVar(value=True)
        self.ref_choice = tk.StringVar(value='')
        self.ref_size = tk.StringVar(value='')
        self.ref_axis = tk.StringVar(value='Cạnh dài')
        self.unit = tk.StringVar(value='mm')
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('TFrame', background='#f3f6fb')
        style.configure('TLabel', background='#f3f6fb', foreground='#17243a')
        style.configure('TLabelframe', background='#f3f6fb')
        style.configure('TLabelframe.Label', background='#f3f6fb', foreground='#17243a', font=('Segoe UI', 10, 'bold'))
        style.configure('TButton', padding=(10, 7))
        style.configure('Accent.TButton', background='#155ed6', foreground='white')
        style.configure('Treeview', rowheight=28, font=('Segoe UI', 10))
        self.build()
        self.panel.canvas.bind('<ButtonPress-1>', self.edit_start)
        self.panel.canvas.bind('<B1-Motion>', self.edit_move)
        self.panel.canvas.bind('<ButtonRelease-1>', self.edit_end)
        for var in (self.blur, self.low, self.high, self.close_kernel, self.min_area, self.max_side, self.exclude_border):
            var.trace_add('write', self.mark_dirty)
        self.method.trace_add('write', self.on_method_change)
        self.update_method_ui()
        self.after(100, self.poll)
        self.bind('<Control-o>', lambda e: self.open_image())
        self.bind('<Control-s>', lambda e: self.save_bundle())
        self.bind('<F5>', lambda e: self.run())

    def build(self):
        header = ttk.Frame(self, padding=(20, 14))
        header.pack(fill='x')
        ttk.Label(header, text='XLA / ĐO KÍCH THƯỚC VẬT THỂ', font=('Segoe UI', 19, 'bold')).pack(anchor='w')
        ttk.Label(header, text='Chọn ảnh · Quan sát từng bước · Hiệu chuẩn · Xuất kết quả').pack(anchor='w', pady=(5, 0))
        toolbar = ttk.Frame(self, padding=(20, 0, 20, 12))
        toolbar.pack(fill='x')
        self.open_btn = ttk.Button(toolbar, text='＋ Chọn ảnh', style='Accent.TButton', command=self.open_image)
        self.open_btn.pack(side='left')
        self.sample_box = ttk.Combobox(toolbar, state='readonly', width=18, values=[p.name for p in sorted((ROOT/'images').glob('*.png'))])
        self.sample_box.set('Mở ảnh mẫu…')
        self.sample_box.pack(side='left', padx=10)
        self.sample_box.bind('<<ComboboxSelected>>', lambda e: self.open_path(ROOT/'images'/self.sample_box.get()))
        self.save_btn = ttk.Button(toolbar, text='Lưu gói kết quả (ZIP)', command=self.save_bundle, state='disabled')
        self.save_btn.pack(side='right')
        self.png_btn = ttk.Button(toolbar, text='Lưu ảnh PNG', command=self.save_png, state='disabled')
        self.png_btn.pack(side='right', padx=8)
        content = ttk.Frame(self, padding=(20, 0, 20, 0))
        content.pack(fill='both', expand=True)
        sidebar = ttk.Frame(content, width=300)
        sidebar.pack(side='left', fill='y', padx=(0, 16))
        sidebar.pack_propagate(False)
        side_canvas = tk.Canvas(sidebar, width=280, bg='#f3f6fb', highlightthickness=0)
        side_scroll = ttk.Scrollbar(sidebar, orient='vertical', command=side_canvas.yview)
        side_scroll.pack(side='right', fill='y')
        side_canvas.pack(side='left', fill='both', expand=True)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side = ttk.Frame(side_canvas)
        side_window = side_canvas.create_window((0, 0), window=side, anchor='nw')
        side.bind('<Configure>', lambda e: side_canvas.configure(scrollregion=side_canvas.bbox('all')))
        side_canvas.bind('<Configure>', lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        settings = ttk.LabelFrame(side, text='1. Tham số xử lý', padding=10)
        settings.pack(fill='x')
        self.settings_frame = settings
        self.parameter_widgets = []

        # Hàng phương pháp (Luôn hiển thị)
        self.row_method = ttk.Frame(settings)
        self.row_method.pack(fill='x', pady=3)
        ttk.Label(self.row_method, text='Phương pháp', font=('Segoe UI', 9)).pack(side='left')
        method_combo = ttk.Combobox(self.row_method, textvariable=self.method,
                                    values=['Canny', 'GrabCut', 'Otsu light', 'Otsu dark'],
                                    width=11, state='readonly')
        method_combo.pack(side='right')
        method_combo.bind('<<ComboboxSelected>>', lambda e: self.update_method_ui())
        self.parameter_widgets.append((method_combo, 'readonly'))

        # Hàng Gaussian (Dùng cho Canny, Otsu)
        self.row_blur = ttk.Frame(settings)
        ttk.Label(self.row_blur, text='Gaussian', font=('Segoe UI', 9)).pack(side='left')
        blur_combo = ttk.Combobox(self.row_blur, textvariable=self.blur,
                                  values=[1, 3, 5, 7, 9, 11, 15], width=11, state='readonly')
        blur_combo.pack(side='right')
        self.parameter_widgets.append((blur_combo, 'readonly'))

        # Hàng Canny thấp (Chỉ dùng cho Canny)
        self.row_low = ttk.Frame(settings)
        ttk.Label(self.row_low, text='Canny thấp', font=('Segoe UI', 9)).pack(side='left')
        low_entry = ttk.Entry(self.row_low, textvariable=self.low, width=13)
        low_entry.pack(side='right')
        self.parameter_widgets.append((low_entry, 'normal'))

        # Hàng Canny cao (Chỉ dùng cho Canny)
        self.row_high = ttk.Frame(settings)
        ttk.Label(self.row_high, text='Canny cao', font=('Segoe UI', 9)).pack(side='left')
        high_entry = ttk.Entry(self.row_high, textvariable=self.high, width=13)
        high_entry.pack(side='right')
        self.parameter_widgets.append((high_entry, 'normal'))

        # Hàng Kernel đóng (Dùng cho Canny, Otsu)
        self.row_close = ttk.Frame(settings)
        ttk.Label(self.row_close, text='Kernel đóng', font=('Segoe UI', 9)).pack(side='left')
        close_combo = ttk.Combobox(self.row_close, textvariable=self.close_kernel,
                                   values=[1, 3, 5, 7, 9], width=11, state='readonly')
        close_combo.pack(side='right')
        self.parameter_widgets.append((close_combo, 'readonly'))

        # Hàng Diện tích min (Chung cho tất cả)
        self.row_min_area = ttk.Frame(settings)
        ttk.Label(self.row_min_area, text='Diện tích min (px²)', font=('Segoe UI', 9)).pack(side='left')
        min_area_entry = ttk.Entry(self.row_min_area, textvariable=self.min_area, width=13)
        min_area_entry.pack(side='right')
        self.parameter_widgets.append((min_area_entry, 'normal'))

        # Hàng Cạnh xử lý tối đa (Chung cho tất cả)
        self.row_max_side = ttk.Frame(settings)
        ttk.Label(self.row_max_side, text='Cạnh tối đa (px)', font=('Segoe UI', 9)).pack(side='left')
        max_side_combo = ttk.Combobox(self.row_max_side, textvariable=self.max_side,
                                      values=[800, 1000, 1400, 2000, 2400], width=11, state='readonly')
        max_side_combo.pack(side='right')
        self.parameter_widgets.append((max_side_combo, 'readonly'))

        # Hàng Bỏ vật sát mép ảnh (Chung cho tất cả)
        self.row_border = ttk.Frame(settings)
        check = ttk.Checkbutton(self.row_border, text='Bỏ vật sát mép ảnh', variable=self.exclude_border)
        check.pack(side='left', pady=2)
        self.parameter_widgets.append((check, 'normal'))

        # Nút Áp dụng / Xử lý lại (F5)
        self.run_btn = ttk.Button(settings, text='Áp dụng / Xử lý lại (F5)', command=self.run)

        # 2. Khung Tương tác GrabCut (CHỈ HIỂN THỊ KHI CHỌN GRABCUT)
        self.edit_frame = ttk.LabelFrame(side, text='2. Chọn & sửa vật (GrabCut)', padding=10)
        self.tool_combo = ttk.Combobox(self.edit_frame, textvariable=self.tool, state='readonly',
                                       values=['Khoanh vùng', 'Giữ vật', 'Loại nền', 'Vẽ đa giác'])
        self.tool_combo.pack(fill='x')
        ttk.Label(self.edit_frame, text='Kéo khoanh vùng → F5. Tô giữ/loại → F5.\nĐa giác: bấm các đỉnh rồi F5.',
                  wraplength=260).pack(anchor='w', pady=5)
        ttk.Checkbutton(self.edit_frame, text='Giữ các vật cũ ngoài vùng chọn', variable=self.keep_existing).pack(anchor='w', pady=(0, 4))
        ttk.Button(self.edit_frame, text='Hoàn tác nét / đỉnh', command=self.undo_edit).pack(fill='x', pady=2)
        ttk.Button(self.edit_frame, text='Xóa vùng chọn', command=self.reset_selection).pack(fill='x', pady=2)
        ttk.Checkbutton(self.edit_frame, text='Tôi đã kiểm tra đúng đường bao', variable=self.confirmed).pack(anchor='w', pady=(4, 0))

        # 3. Khung Hiệu chuẩn kích thước (Luôn hiển thị)
        self.cal_frame = ttk.LabelFrame(side, text='Hiệu chuẩn kích thước', padding=10)
        self.cal_frame.pack(fill='x', pady=(12, 0))
        ttk.Label(self.cal_frame, text='Cách 1: Theo vật chuẩn trên ảnh:').pack(anchor='w')
        self.ref_box = ttk.Combobox(self.cal_frame, textvariable=self.ref_choice, state='readonly', values=[])
        self.ref_box.pack(fill='x', pady=5)
        ttk.Combobox(self.cal_frame, textvariable=self.ref_axis, state='readonly', values=['Cạnh dài', 'Cạnh ngắn']).pack(fill='x')
        ttk.Label(self.cal_frame, text='Kích thước thực:').pack(anchor='w', pady=(6, 3))
        line = ttk.Frame(self.cal_frame)
        line.pack(fill='x')
        ttk.Entry(line, textvariable=self.ref_size, width=12).pack(side='left', fill='x', expand=True)
        ttk.Combobox(line, textvariable=self.unit, state='readonly', values=['mm', 'cm', 'in'], width=5).pack(side='right', padx=(6, 0))
        self.cal_btn = ttk.Button(self.cal_frame, text='Áp dụng vật chuẩn', command=self.calibrate, state='disabled')
        self.cal_btn.pack(fill='x', pady=(8, 4))

        ttk.Separator(self.cal_frame, orient='horizontal').pack(fill='x', pady=8)
        ttk.Label(self.cal_frame, text='Cách 2: Theo thước/đoạn chuẩn:').pack(anchor='w')
        ttk.Button(self.cal_frame, text='✏️ Kéo đoạn chuẩn trên ảnh', command=self.activate_line_tool).pack(fill='x', pady=(3, 3))
        ttk.Button(self.cal_frame, text='Áp dụng đoạn chuẩn đã kéo', command=self.calibrate_line).pack(fill='x', pady=(0, 4))

        self.clear_btn = ttk.Button(self.cal_frame, text='Bỏ hiệu chuẩn → pixel', command=self.clear_calibration, state='disabled')
        self.clear_btn.pack(fill='x', pady=(4, 0))
        ttk.Label(side, text='Vật chuẩn và vật đo phải cùng mặt phẳng.\nChụp gần vuông góc, nền tương phản,\ncác vật tách rời nhau.\n\nCạnh dài/ngắn là cạnh hộp bao xoay;\nkết quả đo chỉ là ước lượng.', wraplength=265, foreground='#52627b', font=('Segoe UI', 9)).pack(anchor='w', pady=12)
        main = ttk.Frame(content)
        main.pack(side='left', fill='both', expand=True)
        top = ttk.Frame(main)
        top.pack(fill='x', pady=(0, 8))
        ttk.Label(top, textvariable=self.summary, font=('Segoe UI', 10, 'bold')).pack(side='left')
        selector = ttk.Combobox(top, textvariable=self.stage, state='readonly', width=22,
                              values=['Kết quả', 'Ảnh gốc', 'Ảnh xám', 'Lọc Gaussian', 'Biên / Nhị phân', 'Đóng hình thái', 'Vùng vật'])
        selector.pack(side='right')
        selector.bind('<<ComboboxSelected>>', lambda e: self.refresh_image())
        self.panel = ImagePanel(main)
        self.panel.pack(fill='both', expand=True)
        ttk.Label(main, text='BẢNG KÍCH THƯỚC  •  Chọn một dòng để đánh dấu vật trên ảnh', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        table_frame = ttk.Frame(main)
        table_frame.pack(fill='x')
        self.table = ttk.Treeview(table_frame, columns=('id', 'role', 'long', 'short', 'unit', 'area', 'perimeter'), show='headings', height=5, selectmode='browse')
        for key, label, width in [('id', 'ID', 40), ('role', 'Vai trò', 80), ('long', 'Dài', 80), ('short', 'Rộng', 80), ('unit', 'Đơn vị', 50), ('area', 'Diện tích bao (đv²)', 130), ('perimeter', 'Chu vi (đv)', 100)]:
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor='center')
        self.table.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        scrollbar.pack(side='right', fill='y')
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind('<<TreeviewSelect>>', self.select_row)
        ttk.Button(main, text='Lưu bảng CSV', command=self.save_csv).pack(anchor='e')
        ttk.Button(main, text='Bỏ vùng đang chọn khỏi kết quả', command=self.remove_region).pack(anchor='e')
        footer = ttk.Frame(self, padding=(20, 10))
        footer.pack(fill='x')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=110)
        self.progress.pack(side='right', padx=(10, 0))
        ttk.Label(footer, textvariable=self.status, wraplength=1050).pack(side='left')

    def on_method_change(self, *_):
        self.update_method_ui()
        if self.result and not self.busy:
            if self.method.get() == 'GrabCut' and self.result.objects:
                self.status.set('Đã bật GrabCut. Kéo khoanh vùng quanh vật cần sửa/thêm rồi bấm F5 (vật cũ bên ngoài được giữ nguyên).')
            else:
                self.mark_dirty()

    def update_method_ui(self, *args):
        method = self.method.get()

        # Ẩn tất cả các hàng tham số để sắp xếp lại theo phương pháp
        for row in (self.row_blur, self.row_low, self.row_high, self.row_close,
                    self.row_min_area, self.row_max_side, self.row_border):
            row.pack_forget()
        self.run_btn.pack_forget()

        if method == 'Canny':
            # Canny: hiển thị các tham số lọc biên tự động
            self.row_blur.pack(fill='x', pady=3)
            self.row_low.pack(fill='x', pady=3)
            self.row_high.pack(fill='x', pady=3)
            self.row_close.pack(fill='x', pady=3)
            self.row_min_area.pack(fill='x', pady=3)
            self.row_max_side.pack(fill='x', pady=3)
            self.row_border.pack(fill='x', pady=4)
            # Ẩn khung tương tác vẽ của GrabCut
            self.edit_frame.pack_forget()
        elif method in ('Otsu light', 'Otsu dark'):
            # Otsu: ẩn Canny thấp, Canny cao (Otsu tự tính ngưỡng)
            self.row_blur.pack(fill='x', pady=3)
            self.row_close.pack(fill='x', pady=3)
            self.row_min_area.pack(fill='x', pady=3)
            self.row_max_side.pack(fill='x', pady=3)
            self.row_border.pack(fill='x', pady=4)
            # Ẩn khung tương tác vẽ của GrabCut
            self.edit_frame.pack_forget()
        elif method == 'GrabCut':
            # GrabCut: chỉ cần diện tích min, cạnh tối đa, bỏ viền
            self.row_min_area.pack(fill='x', pady=3)
            self.row_max_side.pack(fill='x', pady=3)
            self.row_border.pack(fill='x', pady=4)
            # Hiện khung tương tác GrabCut trước khung hiệu chuẩn
            self.edit_frame.pack(fill='x', pady=8, before=self.cal_frame)
            if self.tool.get() not in ('Khoanh vùng', 'Giữ vật', 'Loại nền', 'Vẽ đa giác'):
                self.tool.set('Khoanh vùng')

        self.run_btn.pack(fill='x', pady=(8, 2))

    def activate_line_tool(self):
        self.tool.set('Đoạn chuẩn')
        self.status.set('Chế độ kéo đoạn chuẩn: Kéo một đường thẳng giữa 2 đầu đoạn trên ảnh rồi nhập kích thước thực bên dưới.')

    def mark_dirty(self, *_):
        if self.result and not self.busy:
            self.dirty = True
            self.status.set('Tham số đã thay đổi. Bấm Áp dụng / F5; ảnh và bảng hiện tại vẫn là kết quả lần xử lý trước.')

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title='Chọn ảnh cần đo', filetypes=[('Ảnh', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'), ('Tất cả', '*.*')])
        if path:
            self.open_path(path)

    def open_path(self, path):
        if self.busy:
            return
        self.path = str(path)
        self.result = None
        self.rows = []
        self.calibration = None
        self.selection = {}
        self.reference_points = []
        self.confirmed.set(False)
        self.run()

    def read_settings(self):
        try:
            settings = Settings(method=self.method.get(), blur=int(self.blur.get()),
                                low=int(self.low.get()), high=int(self.high.get()),
                                close=int(self.close_kernel.get()), min_area=float(self.min_area.get()),
                                max_side=int(self.max_side.get()), exclude_border=self.exclude_border.get())
        except ValueError:
            raise ValueError('Các tham số ngưỡng, kernel, diện tích và kích thước phải là số hợp lệ.')
        settings.validate()
        return settings

    def set_busy(self, value):
        self.busy = value
        for widget in (self.open_btn, self.run_btn):
            widget.configure(state='disabled' if value else 'normal')
        self.sample_box.configure(state='disabled' if value else 'readonly')
        for widget, normal in self.parameter_widgets:
            widget.configure(state='disabled' if value else normal)
        state = 'normal' if not value and self.result is not None else 'disabled'
        for widget in (self.save_btn, self.png_btn, self.cal_btn, self.clear_btn):
            widget.configure(state=state)
        if value:
            self.progress.start(12)
        else:
            self.progress.stop()

    def run(self):
        if self.busy:
            return
        if not self.path:
            self.open_image()
            return
        try:
            settings = self.read_settings()
        except ValueError as error:
            messagebox.showerror('Tham số chưa hợp lệ', str(error))
            return
        path = self.path
        selection = copy.deepcopy(self.selection) if settings.method == 'GrabCut' else None
        if selection and self.result and settings.max_side != self.result.settings.max_side:
            messagebox.showinfo('Vùng chọn', 'Xóa vùng chọn trước khi thay kích thước xử lý, rồi chọn lại.')
            return

        keep_old = (
            settings.method == 'GrabCut'
            and self.result is not None
            and bool(self.result.objects)
            and self.keep_existing.get()
        )
        if keep_old and not selection:
            messagebox.showinfo('Chọn vùng GrabCut', 'Hãy kéo khoanh vùng hoặc vẽ đa giác quanh vật cần thêm/sửa trước khi bấm Áp dụng (F5).')
            return

        prev_result = self.result
        if not keep_old:
            self.confirmed.set(False)
            self.reference_points = []
            self.result, self.rows, self.calibration = None, [], None
            self.ref_choice.set('')
            self.ref_box.configure(values=[])
            self.table.delete(*self.table.get_children())
            self.panel.show(None)

        self.summary.set('Đang xử lý…')
        self.set_busy(True)
        self.status.set(f'Đang xử lý {Path(path).name}…')
        def worker():
            try:
                if keep_old:
                    res = apply_grabcut(prev_result, settings, selection, keep_existing=True)
                else:
                    res = process(path, settings, selection)
                self.messages.put(('ok', res))
            except Exception as error:
                self.messages.put(('error', str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == 'ok':
                self.result = payload
                self.dirty = False
                self.confirmed.set(False)
                valid_ids = {o['id'] for o in payload.objects}
                if self.calibration and self.calibration.get('reference_id') is not None and self.calibration['reference_id'] not in valid_ids:
                    self.calibration = None
                    self.ref_choice.set('')
                self.ref_box.configure(values=[str(o['id']) for o in payload.objects])
                self.stage.set('Kết quả')
                self.refresh_rows()
                w, h = payload.original_size
                self.summary.set(f'{len(payload.objects)} vật thể  |  {w} × {h} px  |  {payload.seconds:.2f} s')
                self.status.set(f'{len(payload.objects)} vùng đề xuất. Kiểm tra đường bao trước khi lưu; chưa hiệu chuẩn → pixel.' if payload.objects else 'Kéo khoanh vùng quanh vật rồi bấm F5. Có thể tô giữ vật/loại nền hoặc vẽ đa giác.')
            else:
                if self.result is not None:
                    w, h = self.result.original_size
                    self.summary.set(f'{len(self.result.objects)} vật thể  |  {w} × {h} px')
                    self.status.set('Không tách được vùng chọn; các vật cũ vẫn được giữ nguyên.')
                else:
                    self.summary.set('Không xử lý được ảnh')
                    self.status.set('Xử lý thất bại. Chọn ảnh khác hoặc kiểm tra tham số.')
                messagebox.showerror('Không xử lý được ảnh', payload)
            self.set_busy(False)
        self.after(100, self.poll)

    def refresh_rows(self):
        self.rows = measurements(self.result, **(self.calibration or {}))
        self.table.delete(*self.table.get_children())
        for row in self.rows:
            self.table.insert('', 'end', iid=str(row['id']), values=(row['id'], 'Vật chuẩn' if row['is_reference'] else 'Vật đo',
                              f"{row['long']:.2f}", f"{row['short']:.2f}", row['unit'], f"{row['area']:.2f}", f"{row['perimeter']:.2f}"))
        self.refresh_image()

    def refresh_image(self):
        if self.result is None:
            return
        selected = self.table.selection()
        selected_id = int(selected[0]) if selected else None
        image = annotate(self.result, self.rows, selected_id, self.calibration) if self.stage.get() == 'Kết quả' else self.result.stages[self.stage.get()]
        self.panel.show(image)

    def select_row(self, event=None):
        selection = self.table.selection()
        if selection:
            self.ref_choice.set(selection[0])
        self.refresh_image()

    def calibrate(self):
        if self.result is None or self.busy:
            return
        if self.dirty:
            messagebox.showinfo('Hiệu chuẩn', 'Bấm F5 xử lý lại vùng đã sửa trước.')
            return
        try:
            calibration = dict(reference_id=int(self.ref_choice.get()),
                               reference_size=float(self.ref_size.get().replace(',', '.')),
                               unit=self.unit.get(), reference_axis='long' if self.ref_axis.get() == 'Cạnh dài' else 'short')
            measurements(self.result, **calibration)
        except (ValueError, TypeError) as error:
            messagebox.showerror('Hiệu chuẩn', 'Chọn ID vật chuẩn và nhập kích thước dương hợp lệ.\n'+str(error))
            return
        self.calibration = calibration
        self.refresh_rows()
        self.status.set(f"Đã hiệu chuẩn theo vật #{calibration['reference_id']} • {self.ref_axis.get()} = {calibration['reference_size']:g} {calibration['unit']}. Thay đổi ô nhập cần bấm Áp dụng để có hiệu lực.")

    def clear_calibration(self):
        if self.result is None or self.busy:
            return
        self.calibration = None
        self.refresh_rows()
        self.status.set('Đã bỏ hiệu chuẩn. Kích thước tính bằng pixel ảnh gốc.')

    def can_save(self):
        if self.result is None or self.busy:
            return False
        if self.dirty:
            messagebox.showinfo('Cần xử lý lại', 'Tham số đã thay đổi. Hãy bấm Áp dụng / F5 trước khi lưu.')
            return False
        if not self.rows:
            messagebox.showinfo('Chưa có kết quả', 'Chưa có kết quả đo nào để xuất.')
            return False
        if self.method.get() == 'GrabCut' and not self.confirmed.get():
            messagebox.showinfo('Kiểm tra vùng đo', 'Chế độ GrabCut yêu cầu đánh dấu xác nhận "Tôi đã kiểm tra đúng đường bao" trước khi xuất.')
            return False
        return True

    def image_point(self, event):
        if self.busy or self.result is None:
            return None
        ox, oy = self.panel.origin
        rx, ry = self.panel.ratio
        x, y = (event.x-ox)/rx, (event.y-oy)/ry
        h, w = self.result.stages['Ảnh gốc'].shape[:2]
        return (int(x), int(y)) if 0 <= x < w and 0 <= y < h else None

    def edit_start(self, event):
        p = self.image_point(event)
        if p is None:
            return
        self.drag_start = p
        tool = self.tool.get()
        if tool == 'Đoạn chuẩn':
            self.reference_points = [p, p]
        else:
            if self.method.get() != 'GrabCut':
                self.status.set(f'Phương pháp {self.method.get()} chạy tự động. Để khoanh vùng hoặc vẽ thủ công, hãy chọn GrabCut.')
                self.drag_start = None
                return
            self.dirty = True
            self.confirmed.set(False)
            if tool == 'Khoanh vùng':
                self.selection = {'rect': [*p, *p], 'strokes': []}
            elif tool == 'Vẽ đa giác':
                if 'polygon' not in self.selection:
                    self.selection = {'polygon': []}
                self.selection['polygon'].append(p)
            elif 'rect' in self.selection or 'polygon' in self.selection:
                self.selection.setdefault('strokes', []).append((tool == 'Giữ vật', [p]))
            else:
                self.status.set('Hãy Khoanh vùng hoặc Vẽ đa giác trước khi tô giữ/loại.')
        self.show_edit()

    def edit_move(self, event):
        p = self.image_point(event)
        if p is None or self.drag_start is None:
            return
        tool = self.tool.get()
        if tool == 'Đoạn chuẩn':
            self.reference_points[-1] = p
        elif tool == 'Khoanh vùng':
            self.selection['rect'] = [*self.drag_start, *p]
        elif tool in ('Giữ vật', 'Loại nền') and self.selection.get('strokes'):
            self.selection['strokes'][-1][1].append(p)
        self.show_edit()

    def edit_end(self, event):
        self.edit_move(event)
        self.drag_start = None

    def show_edit(self):
        if self.result is None:
            return
        selected = self.table.selection()
        selected_id = int(selected[0]) if selected else None
        img = annotate(self.result, self.rows, selected_id, self.calibration) if self.result.objects else self.result.stages['Ảnh gốc'].copy()
        rect = self.selection.get('rect')
        if rect:
            cv2.rectangle(img, tuple(rect[:2]), tuple(rect[2:]), (255, 190, 20), 2)
        for keep, points in self.selection.get('strokes', []):
            color = (20, 230, 80) if keep else (240, 60, 60)
            for p in points:
                cv2.circle(img, tuple(p), 4, color, -1)
            for a, b in zip(points, points[1:]):
                cv2.line(img, tuple(a), tuple(b), color, 9)
        if self.selection.get('polygon'):
            pts = self.selection['polygon']
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) >= 3, (255, 190, 20), 2)
            for p in pts:
                cv2.circle(img, tuple(p), 4, (255, 190, 20), -1)
        if len(self.reference_points) == 2:
            cv2.line(img, tuple(self.reference_points[0]), tuple(self.reference_points[1]), (40, 160, 255), 2)
        self.panel.show(img)

    def undo_edit(self):
        if self.busy:
            return
        if self.selection.get('strokes'):
            self.selection['strokes'].pop()
        elif self.selection.get('polygon'):
            self.selection['polygon'].pop()
        self.dirty = bool(self.selection)
        self.confirmed.set(False)
        self.show_edit()

    def reset_selection(self):
        if self.busy:
            return
        self.selection = {}
        self.reference_points = []
        if self.result is not None:
            self.dirty = False
            self.refresh_image()
            self.status.set('Đã xóa vùng khoanh tạm; các vật đã phát hiện vẫn được giữ nguyên.')
        else:
            self.run()

    def calibrate_line(self):
        if self.busy or self.result is None:
            return
        try:
            if self.dirty:
                raise ValueError('Bấm F5 xử lý vùng đã sửa trước khi hiệu chuẩn.')
            calibration = dict(reference_points=copy.deepcopy(self.reference_points),
                               reference_size=float(self.ref_size.get().replace(',', '.')), unit=self.unit.get())
            measurements(self.result, **calibration)
            self.calibration = calibration
            self.refresh_rows()
            self.status.set('Đã hiệu chuẩn bằng đoạn chuẩn; vật chuẩn và vật đo phải cùng mặt phẳng.')
        except ValueError as error:
            messagebox.showerror('Đoạn chuẩn', str(error))

    def save_csv(self):
        if not self.can_save():
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try:
                with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                    writer.writeheader()
                    writer.writerows(self.rows)
            except OSError as error:
                messagebox.showerror('Lưu CSV', str(error))

    def remove_region(self):
        if self.busy or self.result is None or self.dirty:
            return
        selected = self.table.selection()
        if not selected:
            return
        region_id = int(selected[0])
        self.result.objects = [o for o in self.result.objects if o['id'] != region_id]
        if self.calibration and self.calibration.get('reference_id') == region_id:
            self.calibration = None
        self.ref_box.configure(values=[str(o['id']) for o in self.result.objects])
        self.confirmed.set(False)
        self.refresh_rows()
        self.summary.set(f'{len(self.rows)} vùng còn lại')
        self.status.set('Đã bỏ vùng khỏi kết quả; F5 sẽ phát hiện lại các vùng.')

    def save_png(self):
        if not self.can_save():
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=Path(self.result.source).stem+'_result.png', filetypes=[('PNG', '*.png')])
        if path:
            try:
                Image.fromarray(annotate(self.result, self.rows, calibration=self.calibration)).save(path)
                self.status.set('Đã lưu ảnh kết quả: '+path)
            except Exception as error:
                messagebox.showerror('Không lưu được ảnh', str(error))

    def save_bundle(self):
        if not self.can_save():
            return
        path = filedialog.asksaveasfilename(defaultextension='.zip', initialfile=Path(self.result.source).stem+'_results.zip', filetypes=[('ZIP', '*.zip')])
        if path:
            try:
                export_bundle(path, self.result, self.rows, self.calibration)
                self.status.set('Đã lưu ảnh, các bước xử lý, bảng CSV và thông số JSON: '+path)
            except Exception as error:
                messagebox.showerror('Không lưu được kết quả', str(error))


if __name__ == '__main__':
    App().mainloop()
