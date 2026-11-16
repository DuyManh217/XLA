import numpy as np


def otsu_thresholding(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Tìm ngưỡng Otsu và tạo ảnh nhị phân."""
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / image.size
    global_mean = np.dot(np.arange(256), probabilities)
    class_probability = 0.0
    class_mean = 0.0
    best_threshold = 0
    max_variance = -1.0

    for threshold in range(256):
        class_probability += probabilities[threshold]
        class_mean += threshold * probabilities[threshold]
        other_probability = 1.0 - class_probability
        if class_probability == 0 or other_probability == 0:
            continue
        between_variance = (global_mean * class_probability - class_mean) ** 2 / (
            class_probability * other_probability
        )
        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = threshold

    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > best_threshold] = 255
    return binary, best_threshold
