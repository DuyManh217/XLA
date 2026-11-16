"""
Giao diện người dùng đồ họa (Desktop GUI) - BTL Xử Lý Ảnh
Khởi chạy:
    python gui.py
Tính năng:
- Chọn file ảnh tùy ý từ máy tính hoặc dùng ảnh mẫu có sẵn.
- Tự do nhập kích thước vật chuẩn và chọn đơn vị đo (mm, cm, in).
- Bấm "Bắt đầu đo" để xử lý và hiển thị trực quan các đường đo kích thước lên ảnh.
- Bảng tổng kết số liệu chi tiết: Chiều dài, chiều rộng, diện tích, chu vi.
- Hỗ trợ lưu ảnh kết quả và xuất báo cáo file CSV.
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

from io_utils import bgr_to_gray, midpoint, euclidean_dist, order_points
from spatial_filters import gaussian_blur
from morphology import closing, fill_holes
from edge_detection import canny_edge_detection
from contours import (
    find_contours,
    sort_contours,
    contour_area,
    contour_perimeter,
    min_area_rect,
    box_points,
)
from draw import draw_circle, draw_line, draw_polygon
from measurement_utils import ROOT, load_bgr, save_bgr, prepare_measurement


class DimensionEstimatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ thống đo kích thước đối tượng từ ảnh - BTL Xử Lý Ảnh (PTIT)")
        self.root.geometry("1200x780")
        self.root.minsize(1000, 650)

        self.current_image_path = str(ROOT / "images/example_02.png")
        self.result_image = None
        self.results_data = []

        self._build_ui()

        # Nạp ảnh mặc định nếu có
        if os.path.exists(self.current_image_path):
            self._load_and_display_preview(self.current_image_path)

    def _build_ui(self):
        # 1. Bảng điều khiển phía trên (Control Frame)
        ctrl_frame = ttk.LabelFrame(self.root, text="Bảng điều khiển & Thiết lập thông số", padding=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Chọn ảnh
        ttk.Label(ctrl_frame, text="File ảnh:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lbl_file = ttk.Label(ctrl_frame, text=self.current_image_path, width=40, foreground="blue")
        self.lbl_file.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        btn_browse = ttk.Button(ctrl_frame, text="Chọn file ảnh...", command=self._on_browse_file)
        btn_browse.grid(row=0, column=2, padx=5, pady=5)

        # Chọn ảnh mẫu
        ttk.Label(ctrl_frame, text="Ảnh mẫu có sẵn:").grid(row=0, column=3, sticky=tk.W, padx=10, pady=5)
        self.combo_sample = ttk.Combobox(
            ctrl_frame,
            values=["images/example_01.png", "images/example_02.png", "images/example_03.png"],
            width=26,
            state="readonly",
        )
        self.combo_sample.set("images/example_02.png")
        self.combo_sample.grid(row=0, column=4, padx=5, pady=5)
        self.combo_sample.bind("<<ComboboxSelected>>", self._on_select_sample)

        # Kích thước vật chuẩn
        ttk.Label(ctrl_frame, text="Kích thước vật chuẩn:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.ent_width = ttk.Entry(ctrl_frame, width=12)
        self.ent_width.insert(0, "0.955")
        self.ent_width.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Đơn vị đo
        ttk.Label(ctrl_frame, text="Đơn vị:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.combo_unit = ttk.Combobox(ctrl_frame, values=["in", "mm", "cm"], width=8, state="readonly")
        self.combo_unit.set("in")
        self.combo_unit.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Nút Chạy
        self.btn_run = ttk.Button(ctrl_frame, text="▶ BẮT ĐẦU ĐO LƯỜNG", command=self._process_image)
        self.btn_run.grid(row=1, column=4, padx=10, pady=5)
        ttk.Label(ctrl_frame, text="STT vật chuẩn (trái → phải):").grid(row=2, column=0, padx=5)
        self.ent_reference = ttk.Entry(ctrl_frame, width=12)
        self.ent_reference.insert(0, "1")
        self.ent_reference.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(ctrl_frame, text="Nhập cạnh dài/đường kính theo đơn vị đã chọn.").grid(row=2, column=2, columnspan=3)

        # 2. Khung nội dung chính (Ảnh kết quả + Bảng số liệu)
        main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Khung hiển thị ảnh
        img_frame = ttk.LabelFrame(main_paned, text="Hình ảnh kết quả trực quan", padding=5)
        main_paned.add(img_frame, weight=3)

        self.lbl_image_display = ttk.Label(img_frame, anchor=tk.CENTER)
        self.lbl_image_display.pack(fill=tk.BOTH, expand=True)

        # Khung bảng số liệu đo
        table_frame = ttk.LabelFrame(main_paned, text="Bảng dữ liệu đo lường chi tiết", padding=5)
        main_paned.add(table_frame, weight=2)

        # Thanh nút thao tác bảng
        bar_frame = ttk.Frame(table_frame)
        bar_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.lbl_status = ttk.Label(bar_frame, text="Sẵn sàng thực hiện.", font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        btn_csv = ttk.Button(bar_frame, text="💾 Xuất báo cáo CSV...", command=self._export_csv)
        btn_csv.pack(side=tk.RIGHT, padx=5)

        btn_save_img = ttk.Button(bar_frame, text="🖼 Lưu ảnh kết quả...", command=self._save_image)
        btn_save_img.pack(side=tk.RIGHT, padx=5)

        # Cấu hình Bảng Treeview
        cols = ("stt", "role", "width", "height", "area", "perimeter")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=5)
        self.tree.heading("stt", text="STT")
        self.tree.heading("role", text="Vai trò")
        self.tree.heading("width", text="Chiều rộng")
        self.tree.heading("height", text="Chiều dài")
        self.tree.heading("area", text="Diện tích")
        self.tree.heading("perimeter", text="Chu vi")

        self.tree.column("stt", width=50, anchor=tk.CENTER)
        self.tree.column("role", width=140, anchor=tk.CENTER)
        self.tree.column("width", width=120, anchor=tk.E)
        self.tree.column("height", width=120, anchor=tk.E)
        self.tree.column("area", width=130, anchor=tk.E)
        self.tree.column("perimeter", width=130, anchor=tk.E)

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if file_path:
            self.current_image_path = file_path
            self.lbl_file.configure(text=os.path.basename(file_path))
            self._load_and_display_preview(file_path)

    def _on_select_sample(self, event=None):
        sample = str(ROOT / self.combo_sample.get())
        if os.path.exists(sample):
            self.current_image_path = sample
            self.lbl_file.configure(text=sample)
            self._load_and_display_preview(sample)
            self.combo_unit.set("in")
            self.ent_reference.delete(0, tk.END)
            self.ent_reference.insert(0, "1")
            if "example_03" in sample:
                self.ent_width.delete(0, tk.END)
                self.ent_width.insert(0, "3.5")
            else:
                self.ent_width.delete(0, tk.END)
                self.ent_width.insert(0, "0.955")

    def _load_and_display_preview(self, img_path):
        self.results_data.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.result_image = None
        try:
            cv_img = load_bgr(img_path)
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror("Lỗi", str(error))
            return
        if cv_img is not None:
            self._show_on_ui(cv_img)

    def _show_on_ui(self, bgr_img):
        # Chuyển BGR sang RGB và resize vừa màn hình
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((750, 420), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_image_display.configure(image=tk_img)
        self.lbl_image_display.image = tk_img

    def _process_image(self):
        self.btn_run.configure(state="disabled")
        self.results_data.clear()
        self.result_image = None
        for row in self.tree.get_children():
            self.tree.delete(row)
        try:
            self._measure_image()
        except (ValueError, OSError, cv2.error) as error:
            self.lbl_status.configure(text="Chưa đo được ảnh.", foreground="red")
            messagebox.showerror("Lỗi", str(error))
        finally:
            self.btn_run.configure(state="normal")

    def _measure_image(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Lỗi", "Vui lòng chọn file ảnh hợp lệ!")
            return

        try:
            ref_w = float(self.ent_width.get())
        except ValueError:
            messagebox.showerror("Lỗi", "Kích thước vật chuẩn phải là số hợp lệ!")
            return

        unit = self.combo_unit.get()
        reference = int(self.ent_reference.get())
        if not np.isfinite(ref_w) or ref_w <= 0:
            raise ValueError("Kích thước vật chuẩn phải lớn hơn 0 và hữu hạn.")
        self.lbl_status.configure(text="Đang xử lý thuật toán từ scratch...", foreground="blue")
        self.root.update_idletasks()

        image = load_bgr(self.current_image_path)
        gray = bgr_to_gray(image)
        gray = gaussian_blur(gray, ksize=7, sigma=0)
        edged = canny_edge_detection(gray, 50, 100)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edged = closing(edged, kernel)
        edged = fill_holes(edged)

        cnts = find_contours(edged)
        cnts, pixelsPerMetric = prepare_measurement(cnts, ref_w, reference)
        orig = image.copy()
        self.results_data.clear()

        # Xóa dữ liệu cũ trên bảng
        for row in self.tree.get_children():
            self.tree.delete(row)

        RED = (255, 0, 0)
        PINK = (255, 0, 255)
        WHITE = (255, 255, 255)
        count = 0

        for c in cnts:
            if contour_area(c) < 100:
                continue
            count += 1

            rect = min_area_rect(c)
            box = order_points(box_points(rect))
            (tl, tr, br, bl) = box

            tltrX, tltrY = midpoint(tl, tr)
            blbrX, blbrY = midpoint(bl, br)
            tlblX, tlblY = midpoint(tl, bl)
            trbrX, trbrY = midpoint(tr, br)

            dA = euclidean_dist((tltrX, tltrY), (blbrX, blbrY))
            dB = euclidean_dist((tlblX, tlblY), (trbrX, trbrY))

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            c_area_px = contour_area(c)
            c_perim_px = contour_perimeter(c)
            area_real = c_area_px / (pixelsPerMetric**2)
            perim_real = c_perim_px / pixelsPerMetric

            w_val = min(dimA, dimB)
            h_val = max(dimA, dimB)
            role = "Vật chuẩn (Ref)" if count == reference else f"Vật thể #{count}"

            # Lưu vào danh sách
            self.results_data.append({
                "STT": count,
                "Vai trò": role,
                f"Chiều rộng ({unit})": round(float(w_val), 2),
                f"Chiều dài ({unit})": round(float(h_val), 2),
                f"Diện tích ({unit}2)": round(float(area_real), 2),
                f"Chu vi ({unit})": round(float(perim_real), 2),
            })

            # Chèn vào bảng Treeview
            self.tree.insert("", tk.END, values=(
                count,
                role,
                f"{w_val:.2f} {unit}",
                f"{h_val:.2f} {unit}",
                f"{area_real:.2f} {unit}²",
                f"{perim_real:.2f} {unit}",
            ))

            # Vẽ lên ảnh
            cv2.putText(orig, f"#{count}", tuple(box[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            draw_polygon(orig, box, color=(0, 255, 0), thickness=2)
            for x, y in box:
                draw_circle(orig, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

            draw_circle(orig, (int(tltrX), int(tltrY)), radius=5, color=RED, thickness=-1)
            draw_circle(orig, (int(blbrX), int(blbrY)), radius=5, color=RED, thickness=-1)
            draw_circle(orig, (int(tlblX), int(tlblY)), radius=5, color=RED, thickness=-1)
            draw_circle(orig, (int(trbrX), int(trbrY)), radius=5, color=RED, thickness=-1)

            draw_line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), PINK, 2)
            draw_line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), PINK, 2)

            cv2.putText(
                orig,
                f"{dimA:.1f}{unit}",
                (int(tltrX - 15), int(tltrY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2,
            )
            cv2.putText(
                orig,
                f"{dimB:.1f}{unit}",
                (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2,
            )

        self.result_image = orig
        self._show_on_ui(orig)
        self.lbl_status.configure(
            text=f"Hoàn thành: Đo được {count} vật thể (PPM: {pixelsPerMetric:.2f} px/{unit}).",
            foreground="green",
        )

    def _export_csv(self):
        if not self.results_data:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất CSV!")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="ket_qua_do_btl.csv",
        )
        if save_path:
            keys = self.results_data[0].keys()
            try:
                with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results_data)
            except OSError as error:
                messagebox.showerror("Lỗi lưu CSV", str(error))
                return
            messagebox.showinfo("Thành công", f"Đã xuất báo cáo tại:\n{save_path}")

    def _save_image(self):
        if self.result_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh kết quả để lưu!")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
            initialfile="anh_ket_qua_do.png",
        )
        if save_path:
            try:
                save_bgr(save_path, self.result_image)
            except (OSError, ValueError, cv2.error) as error:
                messagebox.showerror("Lỗi lưu ảnh", str(error))
                return
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DimensionEstimatorApp(root)
    root.mainloop()
