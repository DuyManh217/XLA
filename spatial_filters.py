import numpy as np


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tích chập 2D với zero-padding."""
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve_2d yêu cầu ảnh và kernel 2 chiều")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel phải có kích thước lẻ")

    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(
        image.astype(np.float64),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    result = np.zeros(image.shape, dtype=np.float64)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            region = padded[row : row + kh, column : column + kw]
            result[row, column] = np.sum(region * kernel)
    return result


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Lọc trung vị với zero-padding."""
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize phải là số lẻ dương")
    pad = ksize // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")
    result = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            region = padded[row : row + ksize, column : column + ksize]
            result[row, column] = np.median(region)
    return result


def create_gaussian_kernel(ksize, sigma=0):
    """
    Tạo kernel Gaussian dựa trên công thức trong slide.
    :param ksize: Kích thước kernel (số lẻ, ví dụ 7)
    :param sigma: Độ lệch chuẩn. Nếu = 0, tự động tính giống OpenCV.
    """
    if sigma == 0:
        # Công thức tự động tính sigma của OpenCV khi truyền sigma=0
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    # Tạo lưới tọa độ x, y
    ax = np.arange(-(ksize // 2), (ksize // 2) + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Công thức: G(x, y) = e^(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Chuẩn hóa để tổng các phần tử bằng 1 (tránh làm thay đổi độ sáng ảnh)
    kernel = kernel / np.sum(kernel)

    return kernel


def gaussian_blur(img, ksize=7, sigma=0):
    """
    Làm mờ ảnh bằng bộ lọc Gaussian (tương đương cv2.GaussianBlur).
    """
    # 1. Tạo kernel
    kernel = create_gaussian_kernel(ksize, sigma)

    # 2. Thực hiện nhân chập (Sử dụng hàm convolve_2d from scratch)
    # (Nếu không có hàm convolve_2d, xem lại code ở các câu trước, hoặc dùng hàm dưới đây)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Phản chiếu viền để không tạo khung đen giả khi tìm biên.
    padded_img = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect"
    )
    result = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape

    # Tích chập
    for i in range(h):
        for j in range(w):
            region = padded_img[i : i + kh, j : j + kw]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)
