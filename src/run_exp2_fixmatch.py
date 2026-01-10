import tensorflow as tf
from pathlib import Path
import argparse
import time
from config import StableSSLConfig, ExperimentConfig
from experiments.trainers import FixMatchTrainer
from data_loader_tf2 import DataPipeline
from main import prepare_data_paths, setup_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('preprocessed_v2'),
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.95)
    args = parser.parse_args()

    setup_gpu()
    
    config = StableSSLConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.experiment_name = "Exp2_FixMatch"
    
    # Prepare data
    data_paths = prepare_data_paths(args.data_dir, num_labeled=20, num_validation=50) 
    
    # Initialize trainer
    trainer = FixMatchTrainer(config)
    trainer.confidence_threshold = args.threshold
    
    print("\nStarting Experiment 2: FixMatch")
    print(f"Confidence Threshold: {trainer.confidence_threshold}")
    print(f"Labeled samples: {len(data_paths['labeled']['images'])}")
    
    # Run training
    history = trainer.train(data_paths)
    
    # Save results
    exp_config = ExperimentConfig(experiment_name=config.experiment_name, experiment_type="semi-supervised")
    results_dir = exp_config.get_experiment_dir()
    trainer.student_model.save(results_dir / "final_model")
    print(f"Experiment completed. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
