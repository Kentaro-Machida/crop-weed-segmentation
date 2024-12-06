from typing import Any
import albumentations as A 
from src.pkgs.data_classes.config_class import DataAugmentationConfig


class DataTransformBuilder:
    """
    Configファイルにしたがってデータ拡張を行うためのオブジェクトを生成
    ここでは入力前のNormalize処理やToTensor処理は行わない
    """
    def __init__(self, config: DataAugmentationConfig):
        self.config = config

    def __call__(self, **kwargs: Any):
        return self._build_transformer()(**kwargs)

    def _build_transformer(self) -> A.Compose:
        transforms = []
        if self.config.random_brightness_contrast:
            transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        if self.config.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.5))
        if self.config.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
            

        return A.Compose(transforms)


if __name__ == '__main__':
    config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    import numpy as np
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_data = DataAugmentationConfig(**config["experiment"]["model_dataset_config"]["data_augmentation_config"])
    transformer = DataTransformBuilder(config_data)
    
    test_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 3, (256, 256, 1), dtype=np.uint8)
    transformed = transformer(image=test_np, mask=test_mask)
    print(transformed["image"].shape) 
    print(transformed["mask"].shape)
 
