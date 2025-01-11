"""
学習済みモデルを読み込んで予測を行うためのスクリプト
"""

import yaml
import os
import cv2
import logging

from src.pkgs.data_classes.config_class import ExperimentConfig
from src.pkgs.model_datasets.model_dataset_factory import ModelDatasetFactory
from src.utils.data_loads import load_image, LabelConverter
from src.utils.visualize import save_overlayed_image

# ロガーの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_image(output_save_dir: str, input_image_path: str, input_image, mask, pred_mask_2d):
    logger.info(f"Saving prediction results for image: {input_image_path}")

    # Save input image to output directory
    input_image_name = os.path.basename(input_image_path)
    input_image_save_path = os.path.join(output_save_dir, input_image_name)
    logger.debug(f"Saving input image to: {input_image_save_path}")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(input_image_save_path, input_image)

    # Save overlayed ground truth mask
    ground_truth_name = f"{input_image_name}_gt_overlayed.jpg"
    ground_truth_path = os.path.join(output_save_dir, ground_truth_name)
    logger.debug(f"Saving ground truth overlayed image to: {ground_truth_path}")
    save_overlayed_image(
        img=input_image,
        mask=mask,
        save_path=ground_truth_path
    )

    # Save overlayed predicted mask
    predicted_name = f"{input_image_name}_pred_overlayed.jpg"
    predicted_path = os.path.join(output_save_dir, predicted_name)
    logger.debug(f"Saving predicted overlayed image to: {predicted_path}")
    save_overlayed_image(
        img=input_image,
        mask=pred_mask_2d,
        save_path=predicted_path
    )

def predict_and_save_for_cnn(model, dataset):
    logger.info("Starting predictions using CNN model")

    for i, (item, mask) in enumerate(dataset):
        logger.info(f"Processing image {i + 1}/{len(dataset)}")
        mask = mask.numpy()  # size: (height, width)
        tensor = item.unsqueeze(0)
        pred = model(tensor)[0]  # pred: torch.Size([c, height, width])
        pred_mask_2d = pred.argmax(dim=0).numpy()  # pred_2d: numpy.shape ([height, width])
        input_image = load_image(
            path=image_paths[i],
            reseized_height=model_dataset_config.image_height,
            reseized_width=model_dataset_config.image_width
        )

        save_image(output_save_dir, image_paths[i], input_image, mask, pred_mask_2d)
        logger.info(f"Image {i + 1}/{len(dataset)} processed successfully")


def predict_and_save_for_transformer(model, dataset):
    logger.info("Starting predictions using Transformer model")
    height = model_dataset_config.image_height
    width = model_dataset_config.image_width

    for i, batch in enumerate(dataset):
        logger.info(f"Processing image {i + 1}/{len(dataset)}")
        outputs = model(batch)
        logits = outputs.logits
        pred_mask_2d = logits.argmax(dim=1)[0].numpy() # 256x256
        target_tensor = batch["labels"]
        mask = target_tensor.numpy()  # 224x224
        
        # Resize the input image to the same size as the mask
        input_image = load_image(
            path=image_paths[i],
            reseized_height=height,
            reseized_width=width
        )
        pred_mask_2d = cv2.resize(pred_mask_2d, (height, width), interpolation=cv2.INTER_NEAREST)

        save_image(output_save_dir, image_paths[i], input_image, mask, pred_mask_2d)


logger.info("Prediction script started")

with open("./config.yml") as f:
    logger.info("Loading configuration file: ./config.yml")
    config = yaml.safe_load(f)

predict_config_dict = config["predict"]

# 実験ディレクトリのパス
one_model_config = predict_config_dict["one_model_config"]
experiment_dir = os.path.join(predict_config_dict["tracking_dir"], one_model_config["experiment_id"])
run_dir = os.path.join(experiment_dir, one_model_config["run_id"])

config_path = os.path.join(run_dir, "artifacts/config.yml")
model_path = os.path.join(run_dir, "artifacts/best_model.ckpt")

with open(config_path) as f:
    logger.info(f"Loading training configuration from: {config_path}")
    train_config = yaml.safe_load(f)

experiment_config = ExperimentConfig.from_dict(train_config["experiment"])
model_dataset_config = experiment_config.modeldataset_config

# 結果画像保存ディレクトリ
output_save_dir = os.path.join(run_dir, "artifacts/predictions")
os.makedirs(output_save_dir, exist_ok=True)
logger.info(f"Output save directory: {output_save_dir}")

logger.info(f"Experiment directory: {experiment_dir}")
logger.info(f"Run directory: {run_dir}")

label_converter = LabelConverter()
label_dict = label_converter.task_to_label_dict(experiment_config.task)

if __name__ == "__main__":

    logger.info("Setting up ModelDatasetFactory")
    model_dataset_factory = ModelDatasetFactory(
        config=model_dataset_config,
        data_root_path=experiment_config.data_root_path,
        model_dataset_type=experiment_config.modeldataset_type,
        task=experiment_config.task
    )

    model_dataset = model_dataset_factory.create()
    dataset = model_dataset.get_dataset(phase="test")

    logger.info(f"Loaded test dataset with {len(dataset)} items")

    logger.info("Loading model from checkpoint")

    model = model_dataset.get_model().__class__.load_from_checkpoint(
        checkpoint_path=model_path,
        config=model_dataset_config,
        label_dict=label_dict
    ).to("cpu")

    logger.info("Model and dataset prepared. Beginning prediction.")
    image_paths, _ = model_dataset.get_image_mask_paths("test")
    logger.debug(f"Loaded image paths: {image_paths}")

    if experiment_config.modeldataset_type == 'cnn':
        predict_and_save_for_cnn(model=model, dataset=dataset)
    elif experiment_config.modeldataset_type == 'transformer':
        predict_and_save_for_transformer(model=model, dataset=dataset)

    logger.info("Prediction script completed")
