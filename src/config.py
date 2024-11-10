config = {
    "train_parameter": {
        "max_epocks": 10,
        "device": "cpu",
        "batch_size": 1,
        "learning_rate": 0.001,
    },
    # data_root have to include "train", "val", "test" folders
    "data_root": "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/src/pkgs/datasets/test_dataset",
    "task": "all",  # all(3 classes), crop(2 class), plant1d, plant2d(2 class)
    "model_type": "segformer",  # segformer | unet
    "resized_image_height": 256,
    "resized_image_width": 256,
    # Class label of each mask
    "label": {
        "background": 0,
        "crop": 1,
        "weed": 2
    },
    # Used when task is "plant"
    "plant_model": {
        "patch_size": 32,
        "modeltype": "cnn",  # cnn | fcn
    },
    # Used when task is "crop"
    "crop_model": {
        "backborn": "mobilenet_v2",  # mobilenet_v2 | resnet50 | resnet101
    },
    # Used when task is "all"
    "all_model": {
        "backborn": "mobilenet_v2",  # mobilenet_v2 | resnet50 | resnet101
    },
    "segformer_pretrained_model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "data_augmentation": {
        "random_brightness_contrast": True,
        "random_rotate" : True,
        "random_flip": True
    }
}
