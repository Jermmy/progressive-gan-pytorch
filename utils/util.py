import numpy as np
import cv2


def save_result(rows, cols, images, result_file):
    show_images = []
    for r in range(rows):
        row_images = []
        for c in range(cols):
            if (r * cols + c) >= len(images):
                break
            row_images += [(images[r * cols + c] * 255).astype(np.uint8)[:, :, ::-1]]
        row_images = np.concatenate(row_images, axis=1)
        show_images += [row_images]
    show_images = np.concatenate(show_images, axis=0)
    cv2.imwrite(result_file, show_images)