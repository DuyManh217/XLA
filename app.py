"""Run: python app.py. Desktop UI; calculations run on a background thread."""
from pathlib import Path
import queue
import threading
import copy
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
        self.zoom = 1.0
        self.pan = (0, 0)
        self.canvas = tk.Canvas(self, bg='#101a2d', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.schedule)

    def schedule(self, event=None):
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = self.after(60, self.render)

    def show(self, array):
        if array is None:
            self.zoom, self.pan = 1.0, (0, 0)
        self.array = array
        self.render()

    def display_size(self):
        h, w = self.array.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        fit = min(1, max(1, cw-24)/w, max(1, ch-24)/h)
        return max(1, round(w*fit*self.zoom)), max(1, round(h*fit*self.zoom))

    def zoom_at(self, factor, x=None, y=None):
        if self.array is None:
            return
        x = self.canvas.winfo_width()/2 if x is None else x
        y = self.canvas.winfo_height()/2 if y is None else y
        old_x = (x-self.origin[0])/self.ratio[0]
        old_y = (y-self.origin[1])/self.ratio[1]
        self.zoom = min(8, max(1, self.zoom*factor))
        iw, ih = self.display_size()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.pan = (x-old_x*iw/self.array.shape[1]-(cw-iw)/2,
                    y-old_y*ih/self.array.shape[0]-(ch-ih)/2)
        self.render()

    def reset_view(self):
        self.zoom, self.pan = 1.0, (0, 0)
        self.render()

    def pan_by(self, dx, dy):
        self.pan = (self.pan[0]+dx, self.pan[1]+dy)
        self.render()

    def render(self):
        self.pending = None
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.array is None:
            self.canvas.create_text(w/2, h/2, text='Chọn ảnh để bắt đầu', fill='#aab9cf', font=('Segoe UI', 17))
            return
        image = Image.fromarray(self.array).resize(self.display_size(), Image.Resampling.LANCZOS)
        self.origin = ((w-image.width)/2+self.pan[0], (h-image.height)/2+self.pan[1])
        self.ratio = (image.width/self.array.shape[1], image.height/self.array.shape[0])
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(*self.origin, image=self.photo, anchor='nw')


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
        self.method.set('GrabCut')
        self.selection = {}
        self.drag_start = None
        self.pan_start = None
        self.pan_moved = False
        self.tool = tk.StringVar(value='Khoanh vùng')
        self.confirmed = tk.BooleanVar(value=False)
        self.keep_existing = tk.BooleanVar(value=True)
        self.blur = tk.StringVar(value='7')
        self.low = tk.StringVar(value='50')
        self.high = tk.StringVar(value='100')
        self.close_kernel = tk.StringVar(value='3')
        self.min_area = tk.StringVar(value='100')
        self.max_side = tk.StringVar(value='1400')
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
        for var in (self.method, self.blur, self.low, self.high, self.close_kernel, self.min_area, self.max_side, self.exclude_border):
            var.trace_add('write', self.mark_dirty)
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
        self.parameter_widgets = []
        fields = [('Phương pháp', self.method, ['GrabCut', 'Canny', 'Otsu light', 'Otsu dark']),
                  ('Gaussian', self.blur, [1, 3, 5, 7, 9, 11, 15]),
                  ('Canny thấp', self.low, None), ('Canny cao', self.high, None),
                  ('Kernel đóng', self.close_kernel, [1, 3, 5, 7, 9]),
                  ('Diện tích min (px² xử lý)', self.min_area, None),
                  ('Cạnh xử lý tối đa (px)', self.max_side, [800, 1000, 1400, 2000, 2400])]
        for i, (label, var, values) in enumerate(fields):
            ttk.Label(settings, text=label, font=('Segoe UI', 9)).grid(row=i, column=0, sticky='w', pady=4)
            widget = ttk.Combobox(settings, textvariable=var, values=values, width=10, state='readonly') if values else ttk.Entry(settings, textvariable=var, width=12)
            widget.grid(row=i, column=1, sticky='e', padx=(5, 0))
            self.parameter_widgets.append((widget, 'readonly' if values else 'normal'))
        check = ttk.Checkbutton(settings, text='Bỏ vật sát mép ảnh', variable=self.exclude_border)
        check.grid(row=7, columnspan=2, sticky='w', pady=6)
        self.parameter_widgets.append((check, 'normal'))
        self.edit_frame = ttk.LabelFrame(side, text='Chọn và sửa vùng vật', padding=10)
        ttk.Combobox(self.edit_frame, textvariable=self.tool, state='readonly',
                     values=['Khoanh vùng', 'Giữ vật', 'Loại nền', 'Vẽ đa giác']).pack(fill='x')
        ttk.Label(self.edit_frame, text='Kéo khoanh vùng rồi F5. Tô giữ/loại rồi F5. Đa giác: bấm các đỉnh rồi F5.',
                  wraplength=260).pack(anchor='w', pady=5)
        ttk.Checkbutton(self.edit_frame, text='Giữ các vật cũ ngoài vùng chọn', variable=self.keep_existing).pack(anchor='w', pady=(0, 4))
        ttk.Button(self.edit_frame, text='Hoàn tác nét/đỉnh', command=self.undo_selection).pack(fill='x')
        ttk.Button(self.edit_frame, text='Xóa vùng chọn', command=self.reset_selection).pack(fill='x')
        self.run_btn = ttk.Button(settings, text='Áp dụng / Xử lý lại (F5)', command=self.run)
        self.run_btn.grid(row=8, columnspan=2, sticky='ew')
        cal = ttk.LabelFrame(side, text='2. Hiệu chuẩn kích thước', padding=10)
        cal.pack(fill='x', pady=(12, 0))
        ttk.Label(cal, text='Chọn vật chuẩn theo ID trên ảnh:').pack(anchor='w')
        self.ref_box = ttk.Combobox(cal, textvariable=self.ref_choice, state='readonly', values=[])
        self.ref_box.pack(fill='x', pady=5)
        ttk.Combobox(cal, textvariable=self.ref_axis, state='readonly', values=['Cạnh dài', 'Cạnh ngắn']).pack(fill='x')
        ttk.Label(cal, text='Kích thước thực của cạnh đã chọn:').pack(anchor='w', pady=(6, 3))
        line = ttk.Frame(cal)
        line.pack(fill='x')
        ttk.Entry(line, textvariable=self.ref_size, width=12).pack(side='left', fill='x', expand=True)
        ttk.Combobox(line, textvariable=self.unit, state='readonly', values=['mm', 'cm', 'in'], width=5).pack(side='right', padx=(6, 0))
        self.cal_btn = ttk.Button(cal, text='Áp dụng vật chuẩn', command=self.calibrate, state='disabled')
        self.cal_btn.pack(fill='x', pady=(8, 4))
        self.clear_btn = ttk.Button(cal, text='Bỏ hiệu chuẩn → pixel', command=self.clear_calibration, state='disabled')
        self.clear_btn.pack(fill='x')
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
        ttk.Button(top, text='Vừa khung', command=lambda: self.panel.reset_view()).pack(side='right', padx=3)
        ttk.Button(top, text='＋', width=3, command=lambda: self.panel.zoom_at(1.25)).pack(side='right')
        ttk.Button(top, text='－', width=3, command=lambda: self.panel.zoom_at(0.8)).pack(side='right')
        self.panel = ImagePanel(main)
        self.panel.pack(fill='both', expand=True)
        self.panel.canvas.bind('<ButtonPress-1>', self.edit_start)
        self.panel.canvas.bind('<B1-Motion>', self.edit_move)
        self.panel.canvas.bind('<ButtonRelease-1>', self.edit_end)
        self.panel.canvas.bind('<MouseWheel>', lambda e: self.panel.zoom_at(1.25 if e.delta > 0 else 0.8, e.x, e.y))
        self.panel.canvas.bind('<Button-4>', lambda e: self.panel.zoom_at(1.25, e.x, e.y))
        self.panel.canvas.bind('<Button-5>', lambda e: self.panel.zoom_at(0.8, e.x, e.y))
        self.panel.canvas.bind('<ButtonPress-3>', self.view_start)
        self.panel.canvas.bind('<B3-Motion>', self.view_move)
        self.panel.canvas.bind('<ButtonRelease-3>', self.view_end)
        ttk.Label(main, text='BẢNG KÍCH THƯỚC  •  Con lăn: phóng to  •  Chuột phải: chọn vật  •  Kéo chuột phải: di chuyển ảnh', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        table_frame = ttk.Frame(main)
        table_frame.pack(fill='x')
        self.table = ttk.Treeview(table_frame, columns=('id', 'role', 'long', 'short', 'unit', 'area'), show='headings', height=5, selectmode='browse')
        for key, label, width in [('id', 'ID', 45), ('role', 'Vai trò', 95), ('long', 'Cạnh dài', 105), ('short', 'Cạnh ngắn', 105), ('unit', 'Đơn vị', 65), ('area', 'Diện tích đường bao', 175)]:
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor='center')
        self.table.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        scrollbar.pack(side='right', fill='y')
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind('<<TreeviewSelect>>', self.select_row)
        self.confirm_box = ttk.Checkbutton(main, text='Tôi đã kiểm tra đường bao của vật cần đo', variable=self.confirmed)
        self.confirm_box.pack(anchor='w', pady=4)
        ttk.Button(main, text='Bỏ vùng đang chọn', command=self.remove_region).pack(anchor='e')
        self.method.trace_add('write', self.method_changed)
        self.method_changed()
        footer = ttk.Frame(self, padding=(20, 10))
        footer.pack(fill='x')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=110)
        self.progress.pack(side='right', padx=(10, 0))
        ttk.Label(footer, textvariable=self.status, wraplength=1050).pack(side='left')

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
        self.selection = {}
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
        keep_old = (settings.method == 'GrabCut' and self.result is not None
                    and Path(self.result.source) == Path(path).resolve()
                    and bool(self.result.objects) and self.keep_existing.get())
        if keep_old and not selection:
            messagebox.showinfo('Vùng chọn', 'Hãy khoanh vùng vật cần thêm hoặc sửa trước khi bấm F5.')
            return
        prev_result = self.result
        if not keep_old:
            self.result, self.rows, self.calibration = None, [], None
            self.ref_choice.set('')
            self.ref_box.configure(values=[])
            self.table.delete(*self.table.get_children())
            self.panel.show(None)
        self.confirmed.set(False)
        self.summary.set('Đang xử lý…')
        self.set_busy(True)
        self.status.set(f'Đang xử lý {Path(path).name}…')

        def worker():
            try:
                result = (apply_grabcut(prev_result, settings, selection) if keep_old
                          else process(path, settings, selection))
                self.messages.put(('ok', result))
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
                if self.calibration and self.calibration['reference_id'] not in {o['id'] for o in payload.objects}:
                    self.calibration = None
                    self.ref_choice.set('')
                self.ref_box.configure(values=[str(o['id']) for o in payload.objects])
                self.stage.set('Kết quả')
                self.refresh_rows()
                w, h = payload.original_size
                self.summary.set(f'{len(payload.objects)} vật thể  |  {w} × {h} px  |  {payload.seconds:.2f} s')
                self.status.set(f'{len(payload.objects)} vùng tìm được; hãy kiểm tra đường bao.' if payload.objects else 'GrabCut: khoanh vùng quanh vật rồi bấm F5. Chưa có vật: thử điều chỉnh phương pháp hoặc tham số.')
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
                              f"{row['long']:.2f}", f"{row['short']:.2f}", row['unit'], f"{row['area_px2']:.1f}"))
        self.refresh_image()

    def refresh_image(self):
        if self.result is None:
            return
        selected = self.table.selection()
        selected_id = int(selected[0]) if selected else None
        image = annotate(self.result, self.rows, selected_id) if self.stage.get() == 'Kết quả' else self.result.stages[self.stage.get()]
        self.panel.show(image)

    def method_changed(self, *_):
        if self.method.get() == 'GrabCut':
            self.edit_frame.pack(fill='x', pady=8, before=self.cal_btn.master)
        else:
            self.edit_frame.pack_forget()
        self.tool.set('Khoanh vùng')

    def image_point(self, event):
        if self.result is None:
            return None
        ox, oy = self.panel.origin
        rx, ry = self.panel.ratio
        x, y = (event.x-ox)/rx, (event.y-oy)/ry
        h, w = self.result.stages['Ảnh gốc'].shape[:2]
        return (int(x), int(y)) if 0 <= x < w and 0 <= y < h else None

    def view_start(self, event):
        self.pan_start = (event.x, event.y)
        self.pan_moved = False

    def view_move(self, event):
        if self.pan_start is None:
            return
        dx, dy = event.x-self.pan_start[0], event.y-self.pan_start[1]
        if dx or dy:
            self.pan_moved = True
            self.panel.pan_by(dx, dy)
            self.pan_start = (event.x, event.y)

    def view_end(self, event):
        self.pan_start = None
        if self.pan_moved or self.result is None:
            return
        point = self.image_point(event)
        if point is None:
            return
        matches = [obj for obj in self.result.objects
                   if cv2.pointPolygonTest(np.asarray(obj['contour'], np.float32), point, False) >= 0]
        if matches:
            obj = min(matches, key=lambda item: item['area_px2'])
            self.table.selection_set(str(obj['id']))
            self.table.see(str(obj['id']))

    def edit_start(self, event):
        if self.method.get() != 'GrabCut' or self.busy:
            return
        p = self.image_point(event)
        if p is None:
            return
        self.drag_start = p
        self.confirmed.set(False)
        self.dirty = True
        tool = self.tool.get()
        if tool == 'Khoanh vùng':
            self.selection = {'rect': [*p, *p], 'strokes': []}
        elif tool == 'Vẽ đa giác':
            self.selection.setdefault('polygon', []).append(p)
        elif 'rect' in self.selection:
            self.selection.setdefault('strokes', []).append((tool == 'Giữ vật', [p]))
        self.show_selection()

    def edit_move(self, event):
        if self.drag_start is None:
            return
        p = self.image_point(event)
        if p is None:
            return
        tool = self.tool.get()
        if tool == 'Khoanh vùng':
            self.selection['rect'] = [*self.drag_start, *p]
        elif tool in ('Giữ vật', 'Loại nền') and self.selection.get('strokes'):
            self.selection['strokes'][-1][1].append(p)
        self.show_selection()

    def edit_end(self, event):
        self.edit_move(event)
        self.drag_start = None

    def show_selection(self):
        if self.result is None:
            return
        image = self.result.stages['Ảnh gốc'].copy()
        rect = self.selection.get('rect')
        if rect:
            cv2.rectangle(image, tuple(rect[:2]), tuple(rect[2:]), (255,190,20), 2)
        for keep, points in self.selection.get('strokes', []):
            color = (20,220,70) if keep else (240,50,50)
            for p in points:
                cv2.circle(image, tuple(p), 4, color, -1)
        if self.selection.get('polygon'):
            cv2.polylines(image, [np.asarray(self.selection['polygon'],np.int32)], True, (255,190,20), 2)
        self.panel.show(image)

    def undo_selection(self):
        if self.selection.get('polygon'):
            self.selection['polygon'].pop()
        elif self.selection.get('strokes'):
            self.selection['strokes'].pop()
        self.confirmed.set(False)
        self.dirty = True
        self.show_selection()

    def reset_selection(self):
        self.selection = {}
        self.confirmed.set(False)
        self.dirty = True
        self.run()

    def remove_region(self):
        chosen = self.table.selection()
        if not chosen or self.result is None:
            return
        region_id = int(chosen[0])
        self.result.objects = [o for o in self.result.objects if o['id'] != region_id]
        rgb = self.result.stages['Ảnh gốc']
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for obj in self.result.objects:
            cv2.drawContours(mask, [np.asarray(obj['contour'], np.int32)], -1, 255, -1)
        overlay = rgb.copy()
        overlay[mask > 0] = (.55*rgb[mask > 0] + .45*np.array([30, 210, 130])).astype(np.uint8)
        self.result.stages['Vùng vật'] = overlay
        if self.calibration and self.calibration.get('reference_id') == region_id:
            self.calibration = None
        self.confirmed.set(False)
        self.refresh_rows()
        self.status.set(f'Đã bỏ vật #{region_id}. Còn {len(self.result.objects)} vật.')

    def select_row(self, event=None):
        selection = self.table.selection()

        if selection:
            self.ref_choice.set(selection[0])

        self.refresh_image()




    def calibrate(self):
        if self.result is None or self.busy:
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
        return True

    def save_png(self):
        if not self.can_save():
            return
        if not self.rows or not self.confirmed.get():
            messagebox.showinfo('Kiểm tra đường bao', 'Hãy xử lý ảnh và xác nhận đường bao trước khi lưu.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=Path(self.result.source).stem+'_result.png', filetypes=[('PNG', '*.png')])
        if path:
            try:
                Image.fromarray(annotate(self.result, self.rows)).save(path)
                self.status.set('Đã lưu ảnh kết quả: '+path)
            except Exception as error:
                messagebox.showerror('Không lưu được ảnh', str(error))

    def save_bundle(self):
        if not self.can_save():
            return
        if not self.rows or not self.confirmed.get():
            messagebox.showinfo('Kiểm tra đường bao', 'Hãy xử lý ảnh và xác nhận đường bao trước khi lưu.')
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
