# Đo kích thước đối tượng từ ảnh

Ứng dụng desktop Python/Tkinter đo chiều dài, chiều rộng, diện tích bao ngoài
và chu vi đường bao. Đây là bài toán xây dựng ứng dụng xử lý ảnh.
Muốn đo mm/cm/in phải có vật hoặc đoạn chuẩn có kích thước biết trước trong ảnh.
Khi chưa hiệu chuẩn, kết quả chỉ có đơn vị pixel.

## Cài đặt và chạy

Python 3.10 trở lên, có Tkinter:

```bash
pip install -r requirements.txt
python gui.py
```

Có thể chạy bằng `python app.py`. Không cài đồng thời opencv-python và
opencv-python-headless trong cùng môi trường.

## Đo một vật trong ảnh thực tế

1. Chọn ảnh và chọn phương pháp phân vùng (Giao diện sẽ tự động hiển thị đúng các tham số/công cụ của phương pháp đó):
   - **Tự động (Canny / Otsu):** Tự động phát hiện biên/phân ngưỡng toàn bộ các vật thể.
   - **Tương tác (GrabCut):** Hiện khung công cụ khoanh vùng để đo 1 vật cụ thể trên nền phức tạp.
2. Với GrabCut, chọn công cụ **Khoanh vùng**, kéo một hình chữ nhật quanh vật, chừa một ít nền.
3. Bấm **Áp dụng / F5** để tách vật.
4. Nếu sai, chọn **Giữ vật** (nét xanh) hoặc **Loại nền** (nét đỏ), tô lên ảnh rồi F5.
   Nét rộng khoảng 9 pixel trên ảnh xử lý. Có nút hoàn tác nét.
5. Nếu tự động vẫn không đúng, chọn **Vẽ đa giác**, bấm tuần tự các điểm
   quanh đường bao ngoài rồi F5. Nút hoàn tác bỏ đỉnh cuối.
6. Xem **Vùng vật**, **Biên / Nhị phân** và **Kết quả** để kiểm tra.
7. Hiệu chuẩn bằng vật hoặc đoạn chuẩn, sau đó đánh dấu **Tôi đã kiểm tra đúng đường bao**.
8. Lưu PNG, CSV hoặc ZIP.

GrabCut đề xuất một vùng liên thông: ưu tiên vùng có nét giữ vật; nếu chưa có
nét giữ thì chọn vùng lớn nhất. Đây chỉ là đề xuất, không bảo đảm đúng mục tiêu.
Logo, tem và lỗ bên trong không được đếm thành vật riêng khi đã có bao ngoài đúng.
Không có nhận dạng ngữ nghĩa: chương trình không biết vật là ốp, lá hay linh kiện.

## Hiệu chuẩn

- **Theo vật:** chọn ID trong bảng, chọn cạnh dài/ngắn, nhập kích thước thật
  và đơn vị, bấm Áp dụng vật chuẩn.
- **Theo đoạn:** chọn công cụ Đoạn chuẩn, kéo hai đầu một đoạn trên vật chuẩn
  có chiều dài biết trước; nhập chiều dài và đơn vị, bấm Áp dụng đoạn chuẩn đã kéo.
  Đoạn chuẩn có thể nằm ngoài vùng khoanh của vật cần đo.
- Thay số nhập chưa có hiệu lực cho đến khi bấm áp dụng.
- Đổi ảnh hoặc chạy lại phân vùng sẽ xóa hiệu chuẩn. Cần chọn lại.
- Đổi kích thước ảnh xử lý cần xóa vùng chọn rồi chọn lại.

Công thức: pixels_per_unit = chiều dài chuẩn theo pixel / chiều dài thật.
Chiều dài và chu vi chia cho hệ số này; diện tích chia cho bình phương hệ số.
Khi resize, tọa độ contour được quy về ảnh gốc trước khi tính hình học.

## Chế độ tự động và giao diện

- Canny: tìm biên; có thể nhận nhầm vết xước, bóng, họa tiết thành vùng.
- Otsu light/dark: vật sáng trên nền tối hoặc vật tối trên nền sáng.
- Điều chỉnh Gaussian, ngưỡng Canny, kernel đóng, diện tích min và bỏ vùng sát mép.
- Xem ảnh gốc, xám, Gaussian, nhị phân, đóng hình thái, lớp phủ vùng vật.
- Chọn dòng trong bảng để làm nổi bật vùng. Xử lý ảnh chạy nền để GUI phản hồi.
- Số lượng hiển thị là vùng đề xuất; người dùng phải kiểm tra.

## Thuật toán và phạm vi thư viện

Luồng ứng dụng trong pipeline.py dùng OpenCV Gaussian, Canny, morphology,
findContours và minAreaRect để xử lý nhanh; Otsu, chuyển xám, Shoelace và chu vi
dùng các hàm NumPy/Python. GrabCut dùng cv2.grabCut với gợi ý vùng vật/nền.
GrabCut là phần hỗ trợ phân vùng bằng thư viện, không tuyên bố tự cài hoặc
là nội dung đã được dạy trong slide. Không dùng mô hình học sâu hay dịch vụ mạng.

Các module spatial_filters.py, edge_detection.py, morphology.py, contours.py
và draw.py vẫn giữ thuật toán tự cài để học và đối chứng.
benchmark.py đánh giá các thuật toán tự cài này, không phải benchmark toàn bộ GUI.
Nền giao diện và pipeline kế thừa từ phiên bản của thành viên nhóm tại
https://github.com/AnhBop123/BTL-XLA (commit a048bd0).

## Cấu trúc

| File | Vai trò |
|---|---|
| app.py / gui.py | Giao diện và điểm khởi chạy |
| pipeline.py | Xử lý, đo, hiệu chuẩn, xuất ZIP dùng chung |
| segmentation.py | GrabCut, nét hướng dẫn và mặt nạ đa giác |
| main.py | CLI tự động Canny/Otsu |
| test_application.py | Kiểm thử luồng ứng dụng |
| test_algorithms.py | Kiểm thử thuật toán tự cài |
| benchmark.py | Đối chứng thuật toán tự cài với OpenCV |

## Kiểm thử và CLI

```bash
python -m unittest -v
python benchmark.py
python main.py --image images/example_01.png --no-show
python main.py --image images/example_01.png --width 24.26 --reference 1 --unit mm --output data/output/mm
```

CLI không yêu cầu xác nhận thủ công; các vùng xuất ra vẫn phải được người dùng
kiểm tra trên result.png. Không nhập width thì CLI xuất pixel.

## Kết quả và giới hạn

CSV gồm ID, vai trò chuẩn, cạnh dài/ngắn, đơn vị, diện tích, chu vi, diện tích pixel².
ZIP có ảnh kết quả, các ảnh trung gian, CSV và metadata về tham số, số đo,
hiệu chuẩn. Diện tích là diện tích bao ngoài: không trừ lỗ camera hoặc lỗ vật liệu.
Chu vi là chu vi đường bao ngoài, không phải chu vi hộp chữ nhật.

Vật chuẩn và vật đo phải cùng mặt phẳng, chụp gần vuông góc. Hệ số pixel/đơn vị
không thay thế hiệu chuẩn nội tại camera, sửa méo ống kính hoặc biến đổi phối cảnh.
Chưa tự sửa phối cảnh, đo phần bị che khuất, nhận diện lỗ thật hay suy ra kích thước
từ ảnh thiếu chuẩn. Ốp trong suốt, phản sáng, nền giống vật có thể cần vẽ đa giác
hoặc chụp lại. Đường bao thủ công cũng có sai số.

Ảnh tối đa 40 triệu pixel; thu nhỏ có thể mất chi tiết. Chữ số hiển thị không
đại diện độ chính xác. Cần bộ ảnh có số đo thật độc lập để đánh giá sai số mm/cm.
