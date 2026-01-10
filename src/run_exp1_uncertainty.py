import tensorflow as tf
from pathlib import Path
import argparse
import time
from config import StableSSLConfig, ExperimentConfig
from experiments.trainers import UncertaintyMeanTeacherTrainer
from data_loader_tf2 import DataPipeline
from main import prepare_data_paths, setup_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('preprocessed_v2'),
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    setup_gpu()
    
    config = StableSSLConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    # Experiment specific settings
    config.experiment_name = "Exp1_UncertaintyMT"
    
    # Prepare data
    data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=50) 
    
    # Initialize trainer
    trainer = UncertaintyMeanTeacherTrainer(config)
    
    # Visualizing experiment setup
    print("\nStarting Experiment 1: Uncertainty-Aware Mean Teacher")
    print(f"Labeled samples: {len(data_paths['labeled']['images'])}")
    print(f"Unlabeled samples: {len(data_paths['unlabeled']['images'])}")
    
    # Run training
    history = trainer.train(data_paths)
    
    # Save results
    exp_config = ExperimentConfig(experiment_name=config.experiment_name, experiment_type="semi-supervised")
    results_dir = exp_config.get_experiment_dir()
    trainer.model.save(results_dir / "final_model")
    print(f"Experiment completed. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
