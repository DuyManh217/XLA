# Kiểm chứng bản hoàn thiện

Ngày 16/11/2026. Môi trường: Python 3.12, NumPy 2.3.5, Pillow,
OpenCV headless 5.0.0 trên Linux.

## Đã chạy và đạt

`python -m unittest discover -s tests -v`: 11 kiểm thử đạt.

- Vật chữ nhật tổng hợp biết trước kích thước: kiểm tra số đo và hiệu chuẩn.
- Quy đổi pixel khi ảnh bị resize: chênh lệch trong dung sai 3 pixel ở ảnh thử.
- Vật xoay 37°: cạnh đo trong dung sai 3 pixel ở ảnh thử.
- Ảnh trắng/đen trống ở cả Canny và hai chế độ Otsu.
- Đường bao có thứ tự: diện tích hình vuông 100 × 100 đúng 10.000 pixel².
- Giãn/co NumPy so với OpenCV với cùng padding 0.
- Gaussian ảnh đồng nhất: không tạo viền đen giả.
- Đọc đường dẫn tiếng Việt, ảnh xám và alpha.
- Đầu vào hỏng, tham số sai, NaN/Infinity, vật chuẩn không tồn tại.
- Xuất ZIP gồm ảnh, CSV có BOM, JSON và kiểm tra CRC.
- 3 ảnh mẫu có lần lượt 6, 6, 4 contour được nhận; đã xem trực quan
  ảnh kết quả của example_02 và example_03.

Đã kiểm tra cú pháp toàn bộ mã Python của bản mới và chạy CLI tạo gói kết quả.

## Chưa xác nhận

Môi trường thực thi không có kết nối màn hình desktop khả dụng cho Tkinter,
nên chưa chạy thành công luồng tương tác GUI đầu cuối và chưa xác nhận hiển thị
trên Windows. Có cung cấp `tests/gui_smoke.py` để kiểm tra mở ảnh, xử lý nền,
bảng kết quả, hiệu chuẩn, đổi bước xem, lưu và xử lý lại trên máy có desktop.
Hai file BAT dành cho Windows chưa được chạy trên Windows trong lần kiểm chứng này.

Chưa có bộ ảnh chụp thực kèm số đo ground truth để đánh giá độ chính xác mm/cm.
Kiểm thử số học và ảnh mẫu không thay thế đánh giá sai số thực tế.
