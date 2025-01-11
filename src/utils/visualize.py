"""
可視化に関する関数をまとめたモジュール
"""
"""
可視化に関する関数をまとめたモジュール
"""
import numpy as np
import cv2

def overlay_mask_to_image(
        img: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
        ) -> np.ndarray:
    """
    Overlay mask to image with alpha.
    args:
        img: RGB image as np.ndarray. Shape is (H, W, 3).
        mask: Mask as np.ndarray [1, num_classes]. Shape is (H, W).
        alpha: Transparency factor for overlay, range [0, 1].
    returns:
        overlayed_img: Image with mask overlay as np.ndarray. Shape is (H, W, 3).
    """
    # Ensure the input image is in range [0, 255]

    # Define a color map for mask values (1-10)
    color_map = {
        1: (0, 0, 255),    # Blue
        2: (0, 255, 0),    # Green
        3: (255, 0, 0),    # Red
        4: (255, 255, 0),  # Yellow
        5: (255, 0, 255),  # Magenta
        6: (0, 255, 255),  # Cyan
        7: (128, 128, 128),# Gray
        8: (128, 0, 0),    # Maroon
        9: (0, 128, 0),    # Dark Green
        10: (0, 0, 128),   # Navy
    }

    # Create an overlay image of the same shape as the input image
    overlay = np.zeros_like(img, dtype=np.uint8)

    # Assign colors based on mask values
    for value, color in color_map.items():
        overlay[mask == value] = color
    # BGR -> RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


    # Blend the original image and the overlay
    overlayed_img = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)

    return overlayed_img

def save_overlayed_image(
        img: np.ndarray,
        mask: np.ndarray,
        save_path: str,
        alpha: float = 0.5
        ) -> None:
    """
    Overlay mask to image and save the result.
    args:
        img: RGB image as np.ndarray. Shape is (H, W, 3).
        mask: Mask as np.ndarray [1, num_classes]. Shape is (H, W).
        save_path: Path to save the overlayed image.
        alpha: Transparency factor for overlay, range [0, 1].
    returns:
        None
    """
    overlayed_img = overlay_mask_to_image(img, mask, alpha=alpha)
    cv2.imwrite(save_path, overlayed_img)

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # Test overlay_mask_to_image
    img = cv2.imread("src/pkgs/datasets/test_dataset/train/images/IMG_3319_jpg.rf.8db3803bc8d3a8358df8f67e8505e59c.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread("src/pkgs/datasets/test_dataset/train/masks/IMG_3319_jpg.rf.8db3803bc8d3a8358df8f67e8505e59c_mask.png")[:,:,0]
    overlayed_img = overlay_mask_to_image(img, mask, alpha=0.3)
    print(overlayed_img.shape)

    # Test save_overlayed_image
    save_overlayed_image(img, mask, "overlayed_img.jpg", alpha=0.3)
    print("Overlayed image saved as overlayed_img.jpg")
