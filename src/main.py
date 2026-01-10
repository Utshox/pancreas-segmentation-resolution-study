import tensorflow as tf
from pathlib import Path
import argparse
import time
from datetime import datetime
import numpy as np

#mean_teacher
# from visualization import generate_report_visualizations  # Import if you have this module
from config import StableSSLConfig,ExperimentConfig
from train_ssl_tf2n import MixMatchTrainer, StableSSLTrainer,SupervisedTrainer
from data_loader_tf2 import DataPipeline  # Import DataPipeline


def setup_gpu():
    """Setup GPU for training"""
    print("Setting up GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU found")
        return False

def prepare_data_paths(data_dir, num_labeled=20, num_validation=63):
    """Prepare data paths with validation"""
    def get_case_number(path):
        return int(str(path).split('pancreas_')[-1][:3])

    print("Finding data pairs...")
    all_image_paths = []
    all_label_paths = []

    for folder in sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x)):
        folder_no_nii = str(folder).replace('.nii', '')
        img_path = Path(folder_no_nii) / 'image.npy' # Changed filename
        mask_path = Path(folder_no_nii) / 'mask.npy' # Changed filename

        if img_path.exists() and mask_path.exists():
            print(f"Found pair: {img_path}")
            try:
                # Verify files can be loaded
                img = np.load(str(img_path))
                mask = np.load(str(mask_path))
                if img.shape[2] == mask.shape[2]:  # Verify matching dimensions
                    all_image_paths.append(str(img_path))
                    all_label_paths.append(str(mask_path))
                else:
                    print(f"Skipping {img_path} due to dimension mismatch")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    total_samples = len(all_image_paths)
    print(f"\nTotal valid pairs found: {total_samples}")
    
    if total_samples < (num_labeled + num_validation):
        raise ValueError(f"Not enough valid samples. Found {total_samples}, need at least {num_labeled + num_validation}")

    # Create splits
    train_images = all_image_paths[:num_labeled]
    train_labels = all_label_paths[:num_labeled]
    val_images = all_image_paths[-num_validation:]
    val_labels = all_label_paths[-num_validation:]
    unlabeled_images = all_image_paths[num_labeled:-num_validation]

    # Print split information
    print(f"\nData split:")
    print(f"Training (labeled): {len(train_images)} samples")
    print(f"Training (unlabeled): {len(unlabeled_images)} samples")
    print(f"Validation: {len(val_images)} samples")

    # Verify all paths exist
    for path in train_images + train_labels + val_images + val_labels + unlabeled_images:
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

    return {
        'labeled': {
            'images': train_images,
            'labels': train_labels,
        },
        'unlabeled': {
            'images': unlabeled_images
        },
        'validation': {
            'images': val_images,
            'labels': val_labels,
        }
    }
    
# main.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                       help='Path to data directory', 
                       default='/content/drive/MyDrive/Local_contrastive_loss_data/Task07_Pancreas/cropped')
    parser.add_argument('--training_type', type=str, 
                       choices=['supervised', 'mean_teacher', 'mixmatch'],
                       default='supervised', help='Training type')
    parser.add_argument('--experiment_name', type=str, default='pancreas_seg',
                       help='Name for this experiment')
    args = parser.parse_args()

    # Setup
    gpu_available = setup_gpu()
    config = StableSSLConfig()
    
    # Set dimensions
    config.img_size_x = 512
    config.img_size_y = 512
    config.num_channels = 1
    config.batch_size = 8 if gpu_available else 8

    # Create experiment config
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type=args.training_type,
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    # Prepare data paths based on training type
    if args.training_type == 'supervised':
        # Use all labeled data for supervised
        data_paths = prepare_data_paths(args.data_dir, num_labeled=225, num_validation=56)
        trainer = SupervisedTrainer(config)
    else:
        # Use limited labeled data for SSL
        data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=63)
        trainer = StableSSLTrainer(config) if args.training_type == 'mean_teacher' else MixMatchTrainer(config)

    # Print experiment info
    print("\nExperiment Configuration:")
    print(f"Training Type: {args.training_type}")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Learning rate: {trainer.lr_schedule.get_config()}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    
    # Create experiment directory
    exp_dir = experiment_config.get_experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / 'config.txt', 'w') as f:
        f.write(f"Training Type: {args.training_type}\n")
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Timestamp: {experiment_config.timestamp}\n")
        f.write(f"Learning Rate Config: {trainer.lr_schedule.get_config()}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Image Size: {config.img_size_x}x{config.img_size_y}\n")

    # Train
    history = trainer.train(data_paths)
    
    # Save final results
    results_dir = experiment_config.get_experiment_dir()
    np.save(results_dir / 'training_history.npy', history)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()