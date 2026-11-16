# XLA — Ứng dụng đo kích thước vật thể trong ảnh

Chạy `app.py`, chọn ảnh và nhận kết quả trong cùng một cửa sổ.
Ứng dụng desktop Tkinter, chạy tại máy của bạn, không cần web server.

## 1. Chạy trên Windows

Cần Python **3.10 trở lên** có Tkinter; khuyến nghị Python 3.11 hoặc 3.12.
Giải nén toàn bộ ZIP trước khi chạy. Không mở app.py ngay bên trong ZIP.

Cách nhanh:
1. Mở `SETUP_WINDOWS.bat` lần đầu để tạo `.venv` và cài thư viện (cần Internet).
2. Mở `START_APP.bat` để chạy. Những lần sau chỉ cần bước này.

Hoặc mở Terminal trong thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Nếu đã cài các thư viện vào Python đang sử dụng:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Trong PyCharm/IntelliJ, chọn interpreter `.venv\Scripts\python.exe`, rồi Run `app.py`.
Ứng dụng tìm ảnh mẫu theo vị trí file, không phụ thuộc thư mục Terminal hiện tại.
Trên Linux cần Tkinter của hệ điều hành (ví dụ gói `python3-tk`) và màn hình desktop.
Không cài đồng thời nhiều biến thể OpenCV trong cùng venv. Bản này dùng
`opencv-python-headless`: cửa sổ do Tkinter dựng, không dùng `cv2.imshow`.

## 2. Cách sử dụng

1. Bấm **Chọn ảnh** hoặc chọn một **ảnh mẫu**. Ứng dụng tự xử lý ngay.
2. Xem ảnh có hộp bao, ID và cạnh dài × cạnh ngắn. Bảng bên dưới chứa số đo.
3. Chọn danh sách góc trên phải để xem ảnh gốc, ảnh xám, Gaussian,
   biên/nhị phân, đóng hình thái hoặc kết quả.
4. Với ảnh khó, chọn GrabCut, kéo khoanh vùng quanh vật và bấm **Áp dụng / Xử lý lại** (F5). Bật **Giữ các vật cũ ngoài vùng chọn** để sửa hoặc thêm một vật mà vẫn giữ kết quả ở các vùng khác.
   Dùng con lăn hoặc nút **＋/－** để phóng ảnh; kéo chuột phải để di chuyển ảnh. Nhấp chuột phải lên vùng nhận nhầm, rồi bấm **Bỏ vùng đang chọn** để xóa. Nút **Vừa khung** đưa ảnh về kích thước ban đầu.
   Dùng nét giữ vật/nền để sửa, hoặc vẽ đa giác theo đường bao rồi xử lý lại.
5. Để đổi pixel thành mm/cm/in, chọn đúng **ID vật chuẩn**, chọn cạnh dài/ngắn,
   nhập kích thước thực của cạnh đó, chọn đơn vị, bấm **Áp dụng vật chuẩn**.
6. Bấm **Lưu ảnh PNG** hoặc **Lưu gói kết quả (ZIP)**.

Chọn dòng trong bảng để làm nổi bật vật trên ảnh và chọn ID đó trong ô vật chuẩn.
Đây chỉ là chọn ID; cần bấm nút áp dụng để hiệu chuẩn có hiệu lực.
Mỗi lần xử lý lại hoặc đổi ảnh sẽ xóa hiệu chuẩn để tránh dùng nhầm ID.
Thay đổi các ô hiệu chuẩn chưa làm đổi kết quả cho đến khi bấm áp dụng.
Sau khi đổi tham số xử lý, phải xử lý lại trước khi xuất kết quả.

Phím tắt: Ctrl+O chọn ảnh; F5 xử lý lại; Ctrl+S lưu gói ZIP.
Nếu cửa sổ thấp, kéo thanh cuộn bên trái để xem các phần điều khiển còn lại.

## 3. Kết quả đo có ý nghĩa gì?

- **Cạnh dài/ngắn:** hai cạnh của hình chữ nhật xoay diện tích nhỏ nhất bao vật.
  Không phải tên vật, đường kính chính xác, chiều dài đường cong hay kích thước 3D.
- Chưa có vật chuẩn: số đo là **pixel ảnh gốc**, kể cả khi ảnh xử lý đã thu nhỏ.
- Có vật chuẩn: `pixel_per_unit = cạnh chuẩn theo pixel / kích thước thực`;
  `kích thước vật = cạnh theo pixel / pixel_per_unit`.
- **Diện tích:** chọn một trong ba cách ở mục Tính diện tích. Bảng hiện px² khi chưa
  hiệu chuẩn, hoặc đơn vị² khi đã chọn vật chuẩn. Đây không phải diện tích hộp bao xoay.
  Mặt nạ đo cho thấy đúng vùng được tính.
- Nhãn trên ảnh làm tròn 1 chữ số; bảng 2 chữ số; CSV giữ đến 4 chữ số.
  Số chữ số hiển thị không phải cam kết độ chính xác đo.

Không tự giả định kích thước đồng xu trong ảnh mẫu. Hãy biết hoặc đo kích thước
vật chuẩn thực tế trước khi điền. Khi thay đơn vị, phải điền số đo tương ứng
(ví dụ 20 mm và 2 cm), không chỉ đổi tên đơn vị.

## 4. Điều kiện và giới hạn

Chụp gần vuông góc với mặt phẳng đặt vật, đủ sáng, nền tương phản, các vật rời
nhau. Vật chuẩn và vật đo phải trên cùng mặt phẳng, không có chênh lệch độ cao
đáng kể. Dùng vật chuẩn có cạnh đo được hoặc đường kính đã biết.

Đây là xử lý ảnh cổ điển: chưa có nhận dạng tên vật, tách các vật chồng lấp,
hiệu chỉnh phối cảnh hay méo ống kính. Bóng đổ, nền nhiều họa tiết và biên đứt có
thể làm sai/thiếu/thừa vật. Kết quả là ước lượng; chưa đánh giá sai số trên bộ ảnh
đo thực có ground truth. Không thể suy ra mm chính xác từ một ảnh bất kỳ thiếu
thông tin hiệu chuẩn.

Ảnh đầu vào tối đa 40 triệu pixel. Ảnh lớn được thu nhỏ về cạnh xử lý tối đa
(mặc định 1400 px); ảnh kết quả xuất theo kích thước xử lý, số đo pixel quy về ảnh
gốc. Thu nhỏ làm mất chi tiết nhỏ: tăng cạnh xử lý nếu cần.

## 5. Tham số

| Tham số | Ý nghĩa / cách chỉnh |
|---|---|
| Canny | Tìm biên, phù hợp ảnh mẫu nhiều màu |
| Otsu light | Tách vật sáng trên nền tối |
| Otsu dark | Tách vật tối trên nền sáng |
| Gaussian | Kernel lẻ; tăng để giảm nhiễu nhưng có thể mất vật nhỏ; 1 là không làm mờ |
| Canny thấp/cao | Chỉ tác động ở chế độ Canny; cần thấp < cao |
| Kernel đóng | Nối khe hở nhỏ; tăng quá mức có thể nối hai vật với nhau |
| Tính diện tích | Đường bao: Shoelace; Lấp lỗ: vật đặc như lá; Đếm pixel: vùng đã tách như khe nứt, không lấp lỗ |
| Diện tích min | Lọc diện tích contour tính theo pixel² của ảnh xử lý, trước quy đổi |
| Cạnh xử lý tối đa | Tối đa 2400 px; ảnh lớn hơn chạy chậm hơn |
| Bỏ vật sát mép | Loại biên nền và vật bị cắt cụt gần mép ảnh |

Không tìm thấy vật: thử đổi phương pháp, giảm diện tích min hoặc giảm ngưỡng
Canny. Nếu vật bị nối: giảm kernel đóng, giảm blur hoặc chụp tách vật xa nhau.
Với lá, chọn **Lấp lỗ** (ứng dụng gợi ý kernel đóng 5 khi đang ở 3),
tăng kernel đóng từng bước nếu mép lá còn hở,
rồi xem **Mặt nạ đo**. Vùng xanh/gân hoặc chữ bị lấp nhầm phải được loại bỏ
trước khi dùng số đo. Với linh kiện có lỗ thật, dùng **Đường bao** để đo
diện tích bao ngoài, hoặc **Đếm pixel** trên mặt nạ đã tách để trừ lỗ.
Với khe nứt, tách nứt thành vùng trắng bằng Otsu dark hoặc GrabCut rồi dùng
**Đếm pixel**; không dùng lấp lỗ. Mỗi pixel của ảnh xử lý được quy đổi
theo tỷ lệ hai chiều để trả diện tích về px² của ảnh gốc.

## 6. Nội dung xuất

Gói kết quả chứa:
- `result.png`: ảnh có hộp bao và kích thước.
- `stage_0.png` đến `stage_4.png`: ảnh gốc ở độ phân giải xử lý, xám,
  Gaussian, biên/nhị phân, đóng hình thái.
- `measurements.csv`: bảng mở được trong Excel, UTF-8 có BOM.
- `metadata.json`: tên ảnh, kích thước gốc, tham số, hiệu chuẩn, kết quả và thời gian.

CSV: `id`, `is_reference`, `long`, `short`, `unit`, `area_px2`.

## 7. Cấu trúc và thuật toán dùng trong bài

| File | Vai trò |
|---|---|
| app.py | Giao diện Tkinter, luồng xử lý nền, hiển thị và lưu |
| pipeline.py | Đọc ảnh, xử lý, lọc vật, hiệu chuẩn, xuất kết quả |
| segmentation.py | GrabCut có hướng dẫn và vùng đa giác thủ công |
| main.py | Chạy dòng lệnh, không mở cửa sổ khi import |
| spatial_filters.py | Gaussian, kernel và phép nhân lọc bằng NumPy |
| morphology.py | Giãn/co/mở/đóng nhị phân bằng NumPy |
| thresholding.py | Thuật toán Otsu tự cài đặt |
| contours.py | Contour có thứ tự, Shoelace, hộp bao xoay |
| edge_detection.py | Canny OpenCV; Sobel và Laplace qua kernel NumPy |
| io_utils.py | Đọc/ghi ảnh và tiện ích hình học |
| draw.py | Các hàm vẽ tự cài đặt từ bản gốc, giữ để học/đối chiếu |
| legacy_original/ | Các file Python nguyên gốc để đối chiếu, không dùng bởi app mới |
| tests/ | Kiểm thử tự động cho pipeline và kiểm thử GUI tùy chọn |

**Minh bạch thư viện:** bản hoàn thiện vẫn dùng OpenCV cho Canny,
`findContours(RETR_EXTERNAL)`, `minAreaRect`, `boxPoints` và vẽ nhãn/khung.
Gaussian, chuyển xám, Otsu, hình thái học và diện tích Shoelace được cài bằng
NumPy/Python. Pillow xử lý đọc ảnh, EXIF, alpha, resize, PNG và hiển thị.
Đây **không phải** bản 100% thuật toán from-scratch. Nếu giảng viên cấm các hàm
OpenCV tìm contour/hộp bao thì cần thay những hàm này trước khi nộp.

Chế độ GrabCut dùng OpenCV để phân vùng ảnh theo hình chữ nhật và nét hướng dẫn.
Đây là tiện ích của ứng dụng; phần lý thuyết chính vẫn giải thích Gaussian,
phân ngưỡng, hình thái học, contour, Shoelace và đổi pixel sang độ dài.

Thay đổi chính so với code gốc:
- Bỏ đường dẫn ảnh và kích thước chuẩn ghi cứng, bỏ các cửa sổ chờ phím.
- Flood fill cũ trả điểm biên không theo thứ tự, không thể dùng trực tiếp
  Shoelace; bản mới lấy contour theo thứ tự để tính diện tích đúng.
- Hộp bao cũ quét góc bước 2° và làm tròn sớm; dùng minAreaRect và giữ float.
- Sửa lỗi danh sách contour rỗng và nguy cơ trùng góc khi order_points ở 45°.
- Tối ưu lọc/hình thái học, bỏ vòng lặp từng pixel Python; tránh viền đen giả
  do zero padding trong Gaussian.
- Giữ màu RGB thống nhất ở UI; hỗ trợ tên file Unicode, grayscale, EXIF, alpha.
- Chuẩn hóa requirements thành UTF-8 và thêm Pillow còn thiếu.

## 8. Chạy kiểm thử / CLI

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py images/example_02.png --output result.zip
```

Ví dụ CLI hiệu chuẩn (chỉ dùng giá trị này nếu bạn xác nhận đúng kích thước vật chuẩn):

```powershell
python main.py images/example_02.png --reference 1 --size 24.26 --unit mm --output result.zip
```

Kiểm thử giao diện tùy chọn cần desktop:

```powershell
python tests/gui_smoke.py
```

Xem `VALIDATION.md` để biết những phần đã kiểm chứng và giới hạn môi trường thử.
