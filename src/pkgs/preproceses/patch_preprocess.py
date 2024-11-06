from typing import List
from abc import ABC, abstractmethod
import numpy as np


class BasePatchProcess(ABC):
    """
    Base class for patch preprocess.
    """
    def __init__(
            self,
            patch_h: int,
            patch_w: int,
            image_height: int,
            image_widht: int,
            channels=3
            ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.image_height = image_height
        self.image_widht = image_widht
        self.channels = channels

        # image size have to be multiple of patch size
        assert image_height % patch_h == 0, \
            f"image_height:{self.image_height} have to be multiple of \
                patch_h:{self.patch_h}"
        assert image_widht % patch_w == 0, \
            f"image_widht:{self.image_widht} have to be multiple of \
                patch_w:{self.patch_w}"

    @abstractmethod
    def image_to_patches(self, image: np.ndarray) -> List[np.ndarray]:
        pass

    @abstractmethod
    def patches_to_image(self, patches: List[np.ndarray]) -> np.ndarray:
        pass


class PatchProcessor1d(BasePatchProcess):
    """
    Patch preprocess for 1d signal.
    """
    def image_to_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Convert image to some patches.
        input: image(2d or 3d np.ndarray)の画像, patch size
        output: list of 1d ndarray
        """
        if len(image.shape) == 2:
            image = image[:, :, None]
        h, w, _ = image.shape

        # パッチに分割
        patches = [
            image[i:i+self.patch_h, j:j+self.patch_w].flatten()
            for i in range(0, h, self.patch_h)
            for j in range(0, w, self.patch_w)
            ]

        return patches
    
    def patches_to_image(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct patches as an image.
        input: list of 1d ndarray, image size
        output: 2d nparray
        """
        image_2d = np.zeros(
            (self.image_height, self.image_widht, self.channels)
            )
        w_patch_num = self.image_widht / self.patch_w
        for i_h in range(0, self.image_height, self.patch_h):
            for i_w in range(0, self.image_widht, self.patch_w):
                patch_num = int(i_h / self.patch_h * w_patch_num + i_w / self.patch_w)
                patch = patches[patch_num]
                image_2d[i_h:i_h+self.patch_h, i_w:i_w+self.patch_w] = patch.reshape(self.patch_h, self.patch_w, self.channels)
        return image_2d


class PatchProcessor2d(BasePatchProcess):
    """
    Patch preprocess for 2d image.
    """
    def image_to_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Convert image to some patches.
        input: image(2d or 3d np.ndarray)の画像, patch size
        output: list of 1d ndarray
        """
        if len(image.shape) == 2:
            image = image[:, :, None]
        h, w, _ = image.shape
        # パッチに分割
        patches = [image[i:i+self.patch_h, j:j+self.patch_w]
                   for i in range(0, h, self.patch_h)
                   for j in range(0, w, self.patch_w)]

        return patches

    def patches_to_image(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct patches as an image.
        input: list of 2d ndarray, image size
        output: 2d nparray
        """
        image_2d = np.zeros(
            (self.image_height, self.image_widht, self.channels)
            )
        w_patch_num = self.image_widht / self.patch_w
        for i_h in range(0, self.image_height, self.patch_h):
            for i_w in range(0, self.image_widht, self.patch_w):
                patch_num = int(i_h / self.patch_h * w_patch_num + i_w / self.patch_w)
                patch = patches[patch_num]
                image_2d[i_h:i_h+self.patch_h, i_w:i_w+self.patch_w] = patch 
        return image_2d


if __name__ == "__main__":
    """
    Unit test for patch preprocess.
    """
    test_patch_height = 32
    test_patch_width = 32
    test_image_height = 768
    test_image_width = 1024
    test_channls = 3

    image = np.random.rand(test_image_height, test_image_width, test_channls)
    print(f"Input image shape: {image.shape}")

    # Test for PatchProcessor1d
    print("----- Test for PatchProcessor1d -----")
    patch_processor = PatchProcessor1d(
        test_patch_height,
        test_patch_width,
        test_image_height,
        test_image_width,
        test_channls
    )
    patches = patch_processor.image_to_patches(image)
    print(f"Number of patches: {len(patches)}")
    print(f"Shape of patches: {patches[0].shape}")
    image_reconstructed = patch_processor.patches_to_image(patches)
    assert np.allclose(image, image_reconstructed)

    # Test for PatchProcessor2d
    print("----- Test for PatchProcessor2d -----")
    patch_processor = PatchProcessor2d(
        test_patch_height,
        test_patch_width,
        test_image_height,
        test_image_width,
        test_channls
    )

    patches = patch_processor.image_to_patches(image)
    print(f"Number of patches: {len(patches)}")
    print(f"Shape of patches: {patches[0].shape}")
    image_reconstructed = patch_processor.patches_to_image(patches)
    assert np.allclose(image, image_reconstructed)

    print("All test passed.")
