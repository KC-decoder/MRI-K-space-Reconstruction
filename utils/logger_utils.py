import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import wandb

class Logger:
    """
    Custom Logger with optional WandB support.
    """

    def __init__(self, logging_level: str = "INFO", exp_path: Path = None, use_wandb: bool = False):
        """
        Initialize the logger.

        Args:
            logging_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            exp_path (Path): Experiment path where logs are stored
            use_wandb (bool): Whether to log metrics to Weights & Biases (wandb)
        """
        self.logging_level = logging_level
        self.use_wandb = use_wandb

        # Ensure log directory exists
        self.exp_path = exp_path
        self.log_path = self.exp_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Log file named with date and time
        self.log_file = self.log_path / f"KIKI_train_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        # Setup logging
        self._setup_logger()

        # Initialize WandB if needed
        if self.use_wandb:
            wandb.init(project="MRI-Reconstruction", dir=self.exp_path, name=self.exp_path.stem)

    def _setup_logger(self):
        """Set up the logger with a file and console handler."""
        self.logger = logging.getLogger()
        self.logger.setLevel(self._get_level())

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_level())

        # File Handler
        file_handler = logging.FileHandler(self.log_file, mode="a")
        file_handler.setLevel(self._get_level())

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _get_level(self):
        """Convert string logging level to logging module level."""
        return getattr(logging, self.logging_level.upper(), logging.INFO)

    def log(self, message: str, level: str = "INFO"):
        """Log message to console, file, and optionally WandB."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

        # Log to WandB if enabled
        if self.use_wandb:
            wandb.log({"log_message": message})

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log training/evaluation metrics to WandB."""
        if self.use_wandb:
            wandb.log(metrics, step=step)