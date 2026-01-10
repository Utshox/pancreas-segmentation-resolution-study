# In config.py

from dataclasses import dataclass, field
from pathlib import Path
import time

@dataclass
class ExperimentConfig:
    # ... (no changes here) ...
    experiment_name: str
    experiment_type: str  # 'supervised' or 'semi-supervised'
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    results_dir: Path = Path('experiment_results')
    
    def __post_init__(self):
        """Create results directory after initialization"""
        self.results_dir.mkdir(exist_ok=True)
    
    def get_experiment_dir(self):
        """Get and create experiment directory"""
        exp_dir = self.results_dir / f"{self.experiment_type}_{self.experiment_name}_{self.timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

@dataclass
class StableSSLConfig:
    """Configuration optimized for stable SSL training"""
    
    # Data parameters
    img_size_x: int = 512
    img_size_y: int = 512
    num_channels: int = 1
    num_classes: int = 1  # Changed from 2 to 1 for binary segmentation
    batch_size: int = 8 # Reduced batch size for stability
    
    # Model architecture
    initial_filters: int = 32
    n_filters: int = 32  # Added n_filters attribute to match what the model is looking for
    filters: list = field(default_factory=lambda: [32, 64, 128, 256, 512])  # Added filters list
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # Training parameters (can be general or overridden by specific trainers)
    num_epochs: int = 100
    initial_learning_rate: float = 0.00001 # General initial LR, often overridden
    min_learning_rate: float = 1e-6      # General min LR for some schedulers
    warmup_epochs: int = 5               # General warmup epochs for some schedulers
    learning_rate: float = 0.0001        # More specific LR, often set by script args
    weight_decay: float = 1e-5           # General weight decay
    lr_decay_steps: int = 500            # For schedulers like CosineDecayRestarts
    
    # SSL specific parameters (general, can be used by MeanTeacher or others)
    ema_decay: float = 0.999             # Final/Base EMA decay
    consistency_weight: float = 0.1      # General consistency weight, often overridden
    consistency_rampup_epochs: int = 5   # General ramp-up epochs, often overridden by steps
    
    # Loss function parameters
    dice_smooth: float = 1e-6
    temperature: float = 0.5 # General temperature, can be for contrastive or sharpening
    
    # Training stability
    gradient_clip_norm: float = 2.0
    early_stopping_patience: int = 20
    min_delta: float = 0.001
    
    # Augmentation parameters
    noise_std: float = 0.1
    rotation_range: float = 15
    zoom_range: float = 0.1
    scale_range: tuple = (0.9, 1.1)
    brightness_range: tuple = (0.9, 1.1)
    contrast_range: tuple = (0.9, 1.1)
    elastic_deform_sigma: float = 20.0
    elastic_deform_alpha: float = 500.0
    
    # Paths (can be overridden by specific experiment setup)
    checkpoint_dir: Path = Path('ssl_checkpoints_default') # Default, will be overridden
    log_dir: Path = Path('ssl_logs_default')               # Default, will be overridden
    output_dir: Path = Path('ssl_results_default')         # Default, often base for experiment outputs
    experiment_name: str = "DefaultExperiment"             # Default, will be overridden
    
    # --- ADD THESE MIXMATCH SPECIFIC FIELDS ---
    mixmatch_T: float = 0.5
    mixmatch_K: int = 2
    mixmatch_alpha: float = 0.75
    mixmatch_consistency_max: float = 10.0
    mixmatch_rampup_steps: int = 1000 # Rampup in steps for MixMatch consistency

    # --- ADD THESE EMA SPECIFIC FIELDS (can also be used by MeanTeacher if adapted) ---
    initial_ema_decay: float = 0.95
    # ema_decay is already defined above as the final/base EMA decay
    ema_warmup_steps: int = 1000 # Steps for EMA decay to ramp up

    # --- END OF ADDED FIELDS ---

    def __post_init__(self):
        """Create necessary directories after initialization if they use defaults"""
        # Check if paths are still the default Path objects before creating.
        # If they were overridden by strings/Paths during init, this might not be desired.
        # For now, let's assume if output_dir is set, it handles its subdirs.
        if self.output_dir == Path('ssl_results_default'): # Only create if using default
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        
    @property
    def training_steps_per_epoch(self):
        """Calculate training steps per epoch based on dataset size"""
        # This is a placeholder, actual steps_per_epoch should be determined by trainer or args
        return 100
        
    @property
    def consistency_rampup_steps(self): # General consistency ramp-up in steps
        """Calculate total steps for general consistency loss ramp-up"""
        # If mixmatch_rampup_steps is defined, it should take precedence for MixMatch
        return getattr(self, 'mixmatch_rampup_steps', self.consistency_rampup_epochs * self.training_steps_per_epoch)

    @property
    def model_config(self):
        """Get model-specific configuration"""
        return {
            'filters': self.filters,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'num_classes': self.num_classes
        }

    @property
    def training_config(self):
        """Get training-specific configuration"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs
        }

    @property
    def augmentation_config(self):
        """Get data augmentation configuration"""
        return {
            'rotation_range': self.rotation_range,
            'scale_range': self.scale_range,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'elastic_deform_sigma': self.elastic_deform_sigma,
            'elastic_deform_alpha': self.elastic_deform_alpha
        }