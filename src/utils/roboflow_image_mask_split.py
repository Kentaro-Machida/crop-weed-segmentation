"""
Roboflow でアノテーションされたデータを画像とマスクに分割するスクリプト
"""

import os
import shutil
import sys
from pathlib import Path

def copy_and_verify_files(target_dir):
    # 作業ディレクトリの作成
    images_dir = os.path.join(target_dir, "images")
    masks_dir = os.path.join(target_dir, "masks")
    
    if not os.path.exists(target_dir):
        print(f"指定されたディレクトリ {target_dir} は存在しません。")
        sys.exit(1)

    # ディレクトリを作成
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # ファイルのコピー処理
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        
        if "_mask" in file_name:
            shutil.copy(file_path, masks_dir)
        else:
            shutil.copy(file_path, images_dir)

    # 整合性の確認
    image_files = {Path(f).stem for f in os.listdir(images_dir) if f.endswith(".jpg")}
    mask_files = {Path(f).stem.replace("_mask", "") for f in os.listdir(masks_dir) if f.endswith(".png")}

    if image_files != mask_files:
        print("整合性が取れていません。コピーされたファイルを削除します。")
        shutil.rmtree(images_dir)
        shutil.rmtree(masks_dir)
    else:
        print("整合性が取れました。処理を完了しました。")

def sort_and_check_consistncy(img_paths: list, mask_paths: list):
    """
    Sort image and mask paths and check consistency.
    """
    sorted_img_paths = sorted(img_paths)
    sorted_mask_paths = sorted(mask_paths)

    if len(sorted_img_paths) != len(sorted_mask_paths):
        print("Number of images and masks are different. Check the dataset")
        sys.exit(1)

    for img_path, mask_path in zip(sorted_img_paths, sorted_mask_paths):
        img_name = Path(img_path).stem
        mask_name = Path(mask_path).stem.replace("_mask", "")

        if img_name != mask_name:
            print("Image and mask names are different.")
            print(f"Image name: {img_name}")
            print(f"Mask name: {mask_name}")
            sys.exit(1)

    return sorted_img_paths, sorted_mask_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("対象ディレクトリを指定してください。")
        print("使用方法: python script.py <target_dir>")
        sys.exit(1)

    target_dir = sys.argv[1]
    copy_and_verify_files(target_dir)
