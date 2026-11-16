# Các thay đổi và phạm vi

Đối chiếu bản gốc GitHub lehailongt/BTL_XLA tại commit
a673b790d90793ed2d83599bdb65f8c23a39fc82, đã kiểm tra HEAD trực tuyến.
“Bản bạn bổ sung” là trạng thái thư mục trước lần cải thiện này.

| Nội dung | Bản gốc của bạn cùng nhóm | Bản bạn bổ sung | Lần cải thiện này |
|---|---|---|---|
| Luồng chạy | Script, tham số cố định | CLI argparse, GUI Tkinter | Giữ luồng; thêm STT vật chuẩn, kiểm tra lỗi |
| Lọc, hình thái | Gaussian, closing, các hàm lọc khác | Thêm fill_holes | Giữ các hàm; Gaussian phản chiếu viền |
| Canny | Gọi cv2.Canny | Giữ nguyên | Tự cài Canny, giữ hàm cũ để đối chứng |
| Contour | BFS, điểm biên chưa có thứ tự | Giữ nguyên | Giữ BFS; thêm Moore, deque, xử lý rỗng |
| Hình học | Shoelace, quét góc 2 độ, góc nguyên | Thêm chu vi | Biên có thứ tự; góc float, quét 1 độ, tránh trùng góc 45 độ |
| Kết quả | Ảnh hiển thị | CSV, ảnh, bảng GUI | Thêm STT trên ảnh, chọn chuẩn, kiểm tra ghi thất bại |
| Đánh giá | Chưa có benchmark riêng | Benchmark ảnh 01, so W/H | Cả 3 ảnh, W/H/A/P, IoU Canny, số vật không ghép, tham khảo JSON |
| Kiểm thử | Chưa có | Chưa có unit test | 7 test thuật toán tổng hợp |

## Vì sao phải sửa một ít code gốc?

Shoelace và tổng độ dài cạnh chỉ đúng khi các điểm đi tuần tự quanh biên.
BFS tìm tập pixel của một vùng, không bảo đảm thứ tự đó. Sửa đầu ra contour
giúp giữ nguyên công thức diện tích/chu vi. Ví dụ thẻ ảnh 01 trước đây có
chu vi 534.22 in, nay khoảng 11.38 in; diện tích từ 3.47 thành khoảng 6.73 in².

Zero-padding Gaussian tạo cạnh giả gần khung ảnh; closing và fill_holes có
thể biến nó thành một vùng phủ ảnh. Phản chiếu viền giúp ảnh 03 phát hiện
4 vật thay vì một vùng lớn trong lần chạy bản cũ.

Cách chọn góc bằng tổng/hiệu có thể trùng đỉnh ở góc 45 độ.
Xếp bốn góc theo góc quanh tâm giữ đủ bốn đỉnh và các cạnh đối diện.
Không làm tròn tọa độ trước phép đo; chỉ làm tròn khi vẽ.

## Cơ sở kiến thức

- Chương 0, trang PDF 8–11: tự cài thuật toán, test case, đối chứng, phân tích thất bại.
- Chương 2: lọc không gian và xử lý biên ảnh.
- Chương 5: closing và lấp lỗ; giữ cách cài hiện tại.
- Chương 6, trang PDF 48–56: Gaussian → Sobel → NMS → hysteresis.
- Chương 7, trang PDF 8–10: theo dõi biên Moore và điều kiện dừng theo cặp điểm.
- Giữ vòng lặp, hàm và chú thích đơn giản; không thêm ML, deep learning hay framework.

## Những điểm cần hiểu khi bảo vệ

So OpenCV trên cùng mask chỉ đánh giá phần hình học. Sai lệch vật lý còn phụ thuộc
phân vùng, camera và số đo chuẩn. Sai số cạnh chuẩn bằng 0 là do hiệu chuẩn.
PSNR không phải phần trăm chính xác. Với ảnh 03, sai lệch diện tích/chu vi trung
bình 0.505%/0.253% xuất phát từ khác biệt hệ số hiệu chuẩn của hộp bao xấp xỉ.

Ground truth hiện có ít dữ liệu; không tạo số đo thật giả định. Tên “Card Deck Box”
ở ảnh 03 cần kiểm tra lại vì vật trái nhất trông là thẻ nhỏ. Giá trị 3.5 in giữ
để tương thích dữ liệu cũ, không phải chứng nhận kích thước thực.
Các chuỗi tiếng Việt của mã nguồn gốc đúng UTF-8; lỗi đọc ở lượt đánh giá trước
do console/giải mã. Console CLI đã cấu hình UTF-8, README chuyển từ UTF-16 sang UTF-8.

Giới hạn còn lại: GUI đồng bộ, tốc độ Python chậm với ảnh lớn; không tự nhận diện
vật chuẩn; cần nhập STT; không hiệu chỉnh phối cảnh và không tách vật chồng nhau.
Các số đo diện tích là vùng bao ngoài sau fill_holes, không phải diện tích vật liệu
của vật có lỗ. Không bổ sung Otsu vào GUI chỉ để tăng số chức năng.

## Kiểm tra và sử dụng

Chạy từ thư mục dự án: python -m unittest -v, python benchmark.py, python gui.py.
Các test kiểm tra hình chữ nhật/lõm/tròn, vùng tiếp xúc chéo, hộp xoay, đầu vào lỗi,
closing, lấp lỗ và Canny trên hình tổng hợp. Benchmark chạy cả ba ảnh có sẵn.
STT chuẩn bắt đầu từ 1; chiều chuẩn là cạnh dài/đường kính, cùng đơn vị nhập.
Chọn một vật khác thì phải nhập kích thước thật của chính vật đó.

Giữ nguyên draw.py, morphology.py, thresholding.py và ảnh đầu vào.
Không commit/push thay người dùng. Các kết quả cũ trong data/output được giữ;
cần chạy lại chương trình để xuất kết quả mới.
