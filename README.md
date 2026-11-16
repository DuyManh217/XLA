# BÀI TẬP LỚN MÔN XỬ LÝ ẢNH (INT13146)
## ĐỀ TÀI: ĐO KÍCH THƯỚC ĐỐI TƯỢNG TỪ ẢNH (OBJECT SIZE MEASUREMENT)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa:** Công nghệ Thông tin 1 | **Bộ môn:** Khoa học Máy tính  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  

---

## 1. GIỚI THIỆU BÀI TOÁN & MỤC TIÊU
Trong thị giác máy tính và tự động hóa công nghiệp (Machine Vision / Industrial Inspection), việc đo đạc kích thước cơ học của sản phẩm (chiều dài, chiều rộng, diện tích, chu vi) từ ảnh chụp đóng vai trò tối quan trọng.

Hệ thống sử dụng một **Vật chuẩn (Reference Object)** có kích thước vật lý đã biết trước (ví dụ: đồng xu, thẻ ngân hàng, khối chuẩn). Người dùng chọn số thứ tự vật chuẩn theo thứ tự từ trái sang phải; mặc định là vật đầu tiên. Từ đó:
1. Thuật toán tự động phát hiện và đo kích thước pixel của vật chuẩn để tính hệ số tỷ lệ:
   $$\text{PixelsPerMetric} = \frac{\text{Kích thước đo được (pixel)}}{\text{Kích thước vật lý đã biết (in hoặc mm)}}$$
2. Quy đổi kích thước pixel để ước lượng kích thước thực tế của các vật thể còn lại trên cùng mặt phẳng.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```
BTL_XLA/
│── spatial_filters.py   # Lọc không gian: Lọc Gauss, lọc trung vị, tích chập 2D scratch
│── edge_detection.py    # Phát hiện biên: Canny, Sobel, Laplace
│── thresholding.py      # Phân đoạn ảnh: Thuật toán ngưỡng Otsu scratch
│── morphology.py        # Hình thái học: Co, Giãn, Đóng, Mở, Vá lỗ rỗng (Hole filling) scratch
│── contours.py          # BFS gom vùng, Moore theo dõi biên, diện tích, chu vi, hộp bao xoay
│── draw.py              # Thuật toán đồ họa Bresenham tự vẽ đường thẳng, đa giác, hình tròn scratch
│── io_utils.py          # Đọc/ghi ảnh, chuyển Grayscale ITU-R BT.601, khoảng cách Euclid, sắp xếp góc
│── main.py              # Chương trình chính (Hỗ trợ CLI argparse, tính diện tích, xuất báo cáo CSV)
│── benchmark.py         # Script đối chứng định lượng: So sánh thuật toán tự code vs OpenCV
│── gui.py               # Giao diện người dùng đồ họa Desktop GUI (Tkinter)
│── measurement_utils.py # Kiểm tra vật chuẩn, đọc/ghi ảnh và cấu hình console
│── test_algorithms.py   # Kiểm thử tự động các thuật toán
│── requirements.txt     # Thư viện yêu cầu (numpy, opencv-python, pillow)
│── images/              # Dữ liệu ảnh thực nghiệm
│── data/
│   ├── input/           # Dữ liệu đầu vào & file ground_truth.json
│   └── output/          # Tự động lưu ảnh kết quả (result.png) và file báo cáo (report.csv)
└── README.md            # Báo cáo kỹ thuật chi tiết
```

---

## 3. NGUYÊN LÝ THUẬT TOÁN TỰ CÀI ĐẶT (FROM SCRATCH)

1. **Chuyển đổi ảnh mức xám (Grayscale Conversion):**
   Áp dụng công thức chuẩn truyền hình ITU-R BT.601:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

2. **Lọc Gauss làm mịn (Gaussian Blur):**
   Tạo ma trận trọng số Gauss kích thước $K \times K$ với công thức:
   $$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
   Lọc Gaussian bằng tổng tích ảnh và kernel, phản chiếu viền để không tạo khung tối giả. Hàm tích chập tổng quát vẫn dùng zero-padding.

3. **Phát hiện biên Canny:**
   Sau bước Gaussian, tính gradient Sobel, loại bỏ các điểm không cực đại (NMS), áp dụng ngưỡng kép và nối các điểm biên yếu liên thông với biên mạnh.

4. **Làm liền biên bằng Hình thái học (Morphology):**
   - Phép Giãn (*Dilation*): $(A \oplus B)(x, y) = \max_{(i, j) \in B} A(x - i, y - j)$
   - Phép Co (*Erosion*): $(A \ominus B)(x, y) = \min_{(i, j) \in B} A(x + i, y + j)$
   - Phép Đóng (*Closing* = Giãn rồi Co): Giúp hàn gắn các vết nứt và khe hở đứt đoạn trên đường biên.
   - Phép Vá lỗ rỗng (*Hole Filling*): Dùng thuật toán loang BFS từ 4 biên ảnh trên mặt nạ bù để lấp kín mọi lỗ thủng bên trong thân vật thể.

5. **Trích xuất Contours & Hộp bao xoay (Rotated Bounding Box):**
   - BFS 8 lân cận gom vùng liên thông; theo dõi biên Moore lấy các điểm theo thứ tự đi quanh biên (chương 7, trang PDF 9).
   - Sắp xếp các đối tượng theo thứ tự tọa độ từ trái sang phải (`sort_contours`).
   - Quét góc từ 0 đến 89 độ với bước 1 độ (`min_area_rect`) để tìm hộp bao có diện tích nhỏ nhất trong các góc đã xét. Đây là phương pháp xấp xỉ.
   - Sắp xếp 4 góc tuần tự quanh tâm (`order_points`) để xác định các cạnh đối diện, giữ tọa độ thực trong khi đo.

6. **Đo đạc hình học:**
   - Tính 4 trung điểm của 4 cạnh và khoảng cách Euclid nối giữa các cặp trung điểm đối diện ($d_A, d_B$) để xác định chiều dài và chiều rộng.
   - Tính diện tích bằng công thức Shoelace (Gauss's Area Formula):
     $$\text{Area} = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$
   - Tính chu vi bằng tổng khoảng cách Euclid giữa các đỉnh liên tiếp của contour.

---

## 4. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### 4.1. Cài đặt thư viện
Yêu cầu Python 3.9 trở lên (đã kiểm tra bằng Python 3.13):
```bash
pip install -r requirements.txt
```

### 4.2. Khởi chạy Giao diện đồ họa (Desktop GUI)
Khởi chạy giao diện trực quan thân thiện để demo bảo vệ đồ án:
```bash
python gui.py
```
- Bấm nút **"Chọn file ảnh..."** hoặc chọn các ảnh mẫu có sẵn.
- Chọn STT vật chuẩn, nhập kích thước cạnh dài hoặc đường kính của vật chuẩn và chọn đơn vị đo (`in`, `mm`, `cm`).
- Bấm **"▶ BẮT ĐẦU ĐO LƯỜNG"** để quan sát đường đo kích thước trực tiếp trên ảnh và xem bảng dữ liệu.
- Bấm **"Lưu ảnh kết quả"** hoặc **"Xuất báo cáo CSV"** để lưu file.

### 4.3. Chạy bằng dòng lệnh (CLI)
- Chạy với tham số mặc định:
  ```bash
  python main.py
  ```
- Chạy với ảnh tùy ý và đơn vị đo mm:
  ```bash
  python main.py --image images/example_01.png --width 24.26 --unit mm
  ```
- Chạy chế độ chạy ngầm (không hiện cửa sổ GUI từng bước):
  ```bash
  python main.py --image images/example_02.png --width 0.955 --unit in --no-show
  ```
*(Kết quả ảnh đo lường sẽ tự động được lưu tại `data/output/result.png` và bảng số liệu tại `data/output/report.csv`)*.

### 4.4. Chạy kiểm thử Đối chứng định lượng (Benchmark Scratch vs OpenCV)
Chạy script đối chứng tự động để in bảng so sánh độ chính xác giữa code tự viết và thư viện OpenCV:
```bash
python benchmark.py
```

---

## 5. BẢNG KẾT QUẢ ĐỐI CHỨNG ĐỊNH LƯỢNG (BENCHMARK)

Kết quả thực nghiệm trên ba ảnh mẫu (hình học đối chứng trên cùng mask):

| Ảnh | Số vật | PSNR Gaussian (dB) | IoU biên Canny | Sai lệch TB rộng/dài (%) |
|---|---:|---:|---:|---:|
| example_01 | 6 | 50.60 | 0.990 | 0.136 / 0.078 |
| example_02 | 6 | 51.08 | 0.924 | 0.043 / 0.014 |
| example_03 | 4 | 50.26 | 0.992 | 0.192 / 0.086 |

Không quy đổi PSNR thành phần trăm chính xác. Đối chứng cùng mask chỉ kiểm tra
phần hình học; chưa chứng minh độ chính xác phân vùng hay kích thước vật lý.
Tấm thẻ ảnh 01 đo 1.999 × 3.461 in so với số khai báo 2 × 3.5 in:
sai số 0.035% và 1.102%. Hai ảnh còn lại chưa có đủ số đo thật để đánh giá.

## 6. KIỂM THỬ, THAM SỐ VÀ GIỚI HẠN

```bash
python -m unittest -v
python main.py --image images/example_01.png --width 0.955 --unit in --reference 1 --no-show
python main.py --image images/example_01.png --width 24.26 --unit mm --output data/output/mm --no-show
python benchmark.py
python gui.py
```

STT trên ảnh và bảng tính từ trái sang phải sau lọc nhiễu. GUI có ô chọn STT vật chuẩn.
Nhập kích thước cạnh dài của hộp bao hoặc đường kính theo đúng đơn vị đã chọn.
Đổi ảnh mẫu sẽ đưa đơn vị về inch. Dữ liệu ảnh 03 khai báo kích thước chuẩn 3.5 in,
nhưng tên vật chuẩn trong JSON không khớp rõ ràng vật bên trái; cần xác minh bằng thước trước khi sử dụng số đo thực tế.

Canny trong CLI/GUI tự cài Sobel, NMS, ngưỡng kép và nối biên.
OpenCV dùng để đọc/ghi/ghi chữ/hiển thị và đối chứng. Hàm `canny_opencv` phục vụ
đối chứng, không được gọi trong luồng đo của CLI/GUI. Otsu, median, Sobel, Laplace
là các hàm bổ sung chưa có lựa chọn riêng trên GUI.

Vật cần tách rời, không sát mép ảnh, cùng mặt phẳng, nền tương phản, camera gần
vuông góc. Chưa sửa phối cảnh/méo ống kính hoặc tách vật chồng nhau. Lấp lỗ cho
diện tích bao ngoài, không trừ lỗ thật. Hộp bao quét góc bước 1 độ là xấp xỉ.
GUI xử lý đồng bộ nên có thể tạm không phản hồi vài giây; ảnh lớn sẽ chậm hơn.
