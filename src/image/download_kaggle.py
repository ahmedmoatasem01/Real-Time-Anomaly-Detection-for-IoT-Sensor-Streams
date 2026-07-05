import os
import shutil
import kagglehub
from src.utils.logger import get_logger

logger = get_logger("kaggle_download")

def download_and_extract(target_dir="data/raw/vision"):
    logger.info("Downloading real industrial defect dataset via kagglehub...")
    # Download latest version
    path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    logger.info(f"Downloaded to: {path}")

    # We will traverse and copy all 'ok_front' to 'good' and 'def_front' to 'defective'.
    os.makedirs(os.path.join(target_dir, "train", "good"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "train", "defective"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", "good"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", "defective"), exist_ok=True)

    good_count = 0
    defect_count = 0
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, f)
                
                # Determine class based on folder name
                is_defect = "def_front" in root.lower()
                is_good = "ok_front" in root.lower()
                
                if not (is_defect or is_good):
                    continue
                    
                # The dataset might already have train/test splits. Let's check path.
                is_test = "test" in root.lower()
                
                split = "test" if is_test else "train"
                label = "defective" if is_defect else "good"
                
                dest_dir = os.path.join(target_dir, split, label)
                dest_path = os.path.join(dest_dir, f)
                
                shutil.copy2(src_path, dest_path)
                
                if label == "good":
                    good_count += 1
                else:
                    defect_count += 1
                    
    logger.info(f"Copied {good_count} good and {defect_count} defective images to {target_dir}")

if __name__ == "__main__":
    download_and_extract()
