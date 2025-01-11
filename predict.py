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

def predict(predict_config_dict: dict):
    logger.info("Prediction script started")

    # 実験ディレクトリのパス
    one_model_config = predict_config_dict["one_model_config"]
    experiment_dir = os.path.join(predict_config_dict["tracking_dir"], one_model_config["experiment_id"])
    run_dir = os.path.join(experiment_dir, one_model_config["run_id"])
    config_path = os.path.join(run_dir, "artifacts/config.yml")
    model_path = os.path.join(run_dir, "artifacts/best_model.ckpt")

    # 結果画像保存葉ディレクトリ
    output_save_dir = os.path.join(run_dir, "artifacts/predictions")
    os.makedirs(output_save_dir, exist_ok=True)
    logger.info(f"Output save directory: {output_save_dir}")

    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Loading training configuration from: {config_path}")
    logger.info(f"Loading model checkpoint from: {model_path}")

    with open(config_path) as f:
        train_config = yaml.safe_load(f)
    
    experiment_config = ExperimentConfig.from_dict(train_config["experiment"])
    model_dataset_config = experiment_config.modeldataset_config
    label_converter = LabelConverter()
    label_dict = label_converter.task_to_label_dict(experiment_config.task)

    logger.info("Setting up ModelDatasetFactory")
    model_dataset_factory = ModelDatasetFactory(
        config=model_dataset_config,
        data_root_path=experiment_config.data_root_path,
        model_dataset_type=experiment_config.modeldataset_type,
        task=experiment_config.task
    )
    model_dataset = model_dataset_factory.create()

    logger.info("Retrieving test dataset")
    dataset = model_dataset.get_dataset(phase="test")
    
    logger.info("Loading model from checkpoint")
    model = model_dataset.get_model().__class__.load_from_checkpoint(
        checkpoint_path=model_path,
        config=model_dataset_config,
        label_dict=label_dict
    ).to("cpu")

    logger.info("Model and dataset prepared. Beginning prediction.")
    image_paths, _ = model_dataset.get_image_mask_paths("test")
    
    for i, (tensor, mask) in enumerate(dataset):
        logger.info(f"Processing image {i + 1}/{len(dataset)}")
        tensor = tensor.unsqueeze(0)
        mask_2d = mask.argmax(dim=0).numpy()  # mask_2d: torch.Size([224, 224])
        pred = model(tensor)[0]  # pred: torch.Size([2, 224, 224])
        pred_mask_2d = pred.argmax(dim=0)  # pred_2d: torch.Size([224, 224])
        input_image = load_image(
            path=image_paths[i],
            reseized_height=model_dataset_config.image_height,
            reseized_width=model_dataset_config.image_width
        )

        # Save input image to output directory
        input_image_name = image_paths[i].split("/")[-1]
        input_image_path = os.path.join(output_save_dir, input_image_name)
        logger.info(f"Saving input image to: {input_image_path}")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(input_image_path, input_image)
        
        # Save overlayed ground truth mask
        ground_truth_name = image_paths[i].split("/")[-1] + "_gt_overlayed.jpg"
        ground_truth_path = os.path.join(output_save_dir, ground_truth_name)
        logger.info(f"Saving ground truth overlayed image to: {ground_truth_path}")
        save_overlayed_image(
            img=input_image,
            mask=mask_2d,
            save_path=ground_truth_path
        )

        # Save overlayed predicted mask
        predicted_name = image_paths[i].split("/")[-1] + "_pred_overlayed.jpg"
        predicted_path = os.path.join(output_save_dir, predicted_name)
        logger.info(f"Saving predicted overlayed image to: {predicted_path}")
        save_overlayed_image(
            img=input_image,
            mask=pred_mask_2d,
            save_path=predicted_path
        )

    logger.info("Prediction script finished successfully.")

if __name__ == "__main__":
    with open("./config.yml") as f:
        logger.info("Loading configuration file: ./config.yml")
        config = yaml.safe_load(f)
    predict_config_dict = config["predict"]
    predict(predict_config_dict)
