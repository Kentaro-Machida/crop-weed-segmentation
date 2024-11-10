from typing import Any
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from pkgs.data_classes.config_class import ConfigData


class DataTransformBuilder:
    """
    Configファイルにしたがってデータ拡張を行うためのtransformerを生成
    使用例
    from src.config import config
    import numpy as np
    config_data = ConfigData.load_data(config)
    transformer = DataTransformBuilder(config_data)
    
    test_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 3, (256, 256, 1), dtype=np.uint8)
    transformed = transformer(image=test_np, mask=test_mask)
    """
    def __init__(self, config: ConfigData):
        self.config = config

    def __call__(self, **kwargs: Any):
        return self._build_transformer()(**kwargs)

    def _build_transformer(self) -> A.Compose:
        data_transform_config = self.config.data_augmentation
        transforms = []
        if data_transform_config.random_brightness_contrast:
            transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        if data_transform_config.random_rotate:
            transforms.append(A.Rotate(limit=35, p=0.5))
        if data_transform_config.random_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
            transforms.append(A.VerticalFlip(p=0.5))
        if self.config.model_type != "segformer":
            transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
            transforms.append(ToTensorV2())

        return A.Compose(transforms)


if __name__ == '__main__':
    from src.config import config
    import numpy as np
    config_data = ConfigData.load_data(config)
    transformer = DataTransformBuilder(config_data)
    
    test_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 3, (256, 256, 1), dtype=np.uint8)
    transformed = transformer(image=test_np, mask=test_mask)
    print(transformed["image"].shape) 
    print(transformed["mask"].shape)
 
