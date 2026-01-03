"""
LSTM (Flow and Snow Hydrological LSTM) model runner.

An LSTM-based model for hydrological predictions, specifically for streamflow
and snow water equivalent (SWE).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from torch.utils.data import TensorDataset, DataLoader

from ..registry import ModelRegistry
from ..base import BaseModelRunner
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)

from .model import LSTMModel
from .preprocessor import LSTMPreprocessor
from .postprocessor import LSTMPostprocessor


@ModelRegistry.register_runner('LSTM', method_name='run_lstm')
class LSTMRunner(BaseModelRunner):
    """
    LSTM: Flow and Snow Hydrological LSTM Runner.

    Orchestrates the LSTM model workflow: data loading, preprocessing,
    model training (or loading), simulation, and postprocessing.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized LSTM runner with device: {self.device}")
        
        # Initialize components
        self.preprocessor = LSTMPreprocessor(
            self.config_dict, 
            self.logger, 
            self.project_dir, 
            self.device
        )
        self.postprocessor = LSTMPostprocessor(
            self.config_dict, 
            self.logger, 
            self.project_dir
        )
        
        self.model = None

    def _get_model_name(self) -> str:
        return "LSTM"

    def run_lstm(self):
        """Run the complete LSTM model workflow."""
        self.logger.info("Starting LSTM model run")

        with symfluence_error_handler(
            "LSTM model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # 1. Load Data
            forcing_df, streamflow_df, snow_df = self.preprocessor.load_data()
            
            # Check if snow data should be used based on config
            use_snow = self.config_dict.get('LSTM_USE_SNOW', self.config_dict.get('FLASH_USE_SNOW', False))
            snow_df_input = snow_df if use_snow else pd.DataFrame() # Use empty DF if not using snow

            # 2. Preprocess Data
            # Decide if we are training (fit scalers) or just simulating (load scalers)
            load_existing_model = self.config_dict.get('LSTM_LOAD', self.config_dict.get('FLASH_LOAD', False))
            model_save_path = self.project_dir / 'models' / 'lstm_model.pt'
            
            if load_existing_model:
                # Load pre-trained model state and scalers first
                self.logger.info("Loading pre-trained LSTM model")
                checkpoint = self._load_model_checkpoint(model_save_path)
                
                # Set scalers in preprocessor from checkpoint
                self.preprocessor.set_scalers(
                    checkpoint['feature_scaler'], 
                    checkpoint['target_scaler'],
                    checkpoint['output_size'],
                    checkpoint['target_names']
                )
                
                # Preprocess data using loaded scalers
                X_tensor, y_tensor, common_dates, features_avg = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=False
                )
                
                # Create model structure
                input_size = X_tensor.shape[2]
                self._create_model_instance(
                    input_size, 
                    checkpoint['output_size'],
                    hidden_size=self.config_dict.get('LSTM_HIDDEN_SIZE', self.config_dict.get('FLASH_HIDDEN_SIZE', 64)),
                    num_layers=self.config_dict.get('LSTM_NUM_LAYERS', self.config_dict.get('FLASH_NUM_LAYERS', 2))
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            else:
                # Training mode: Fit scalers
                X_tensor, y_tensor, common_dates, features_avg = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=True
                )
                
                input_size = X_tensor.shape[2]
                hidden_size = self.config_dict.get('LSTM_HIDDEN_SIZE', self.config_dict.get('FLASH_HIDDEN_SIZE', 64))
                num_layers = self.config_dict.get('LSTM_NUM_LAYERS', self.config_dict.get('FLASH_NUM_LAYERS', 2))
                output_size = self.preprocessor.output_size
                
                # Create and Train
                self._create_model_instance(input_size, output_size, hidden_size, num_layers)
                
                self._train_model(
                    X_tensor, 
                    y_tensor,
                    epochs=self.config_dict.get('LSTM_EPOCHS', self.config_dict.get('FLASH_EPOCHS', 100)),
                    batch_size=self.config_dict.get('LSTM_BATCH_SIZE', self.config_dict.get('FLASH_BATCH_SIZE', 32)),
                    learning_rate=self.config_dict.get('LSTM_LEARNING_RATE', self.config_dict.get('FLASH_LEARNING_RATE', 0.001))
                )
                
                # Save model
                self.project_dir.joinpath('models').mkdir(exist_ok=True)
                self._save_model_checkpoint(model_save_path)

            # 3. Simulate (Run Inference on all data)
            results = self._simulate(X_tensor, common_dates, features_avg)

            # 4. Postprocess
            self.postprocessor.visualize_results(results, streamflow_df, snow_df_input, use_snow)
            self.postprocessor.save_results(results, use_snow)

            self.logger.info("LSTM model run completed successfully")

    def _create_model_instance(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        """Create the LSTM model instance."""
        dropout_rate = float(self.config_dict.get('LSTM_DROPOUT', self.config_dict.get('FLASH_DROPOUT', 0.2)))
        self.logger.info(
            f"Creating LSTM model with input_size: {input_size}, hidden_size: {hidden_size}, "
            f"num_layers: {num_layers}, output_size: {output_size}"
        )
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(self.device)

    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, learning_rate: float):
        """Train the LSTM model."""
        self.logger.info(
            f"Training LSTM model with {epochs} epochs, batch_size: {batch_size}, learning_rate: {learning_rate}"
        )

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=float(self.config_dict.get('LSTM_L2_REGULARIZATION', self.config_dict.get('FLASH_L2_REGULARIZATION', 1e-6)))
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience = self.config_dict.get('LSTM_LEARNING_PATIENCE', self.config_dict.get('FLASH_LEARNING_PATIENCE', 20))
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss encountered in epoch {epoch}, batch {i // batch_size}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        self.logger.info("LSTM model training completed")

    def _simulate(self, X_tensor: torch.Tensor, common_dates: pd.DatetimeIndex, features_avg: pd.DataFrame) -> pd.DataFrame:
        """Run full simulation with LSTM model."""
        self.logger.info("Running full simulation with LSTM model")
        
        self._log_memory_usage()

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_predictions = self.model(batch[0])
                predictions.append(batch_predictions.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # Inverse transform the predictions
        predictions = self.preprocessor.target_scaler.inverse_transform(predictions)

        # Handle NaN values
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e15, neginf=-1e15)

        # Create column names based on number of targets
        if self.preprocessor.output_size == 2:
            columns = ['predicted_streamflow', 'predicted_SWE']
        else:
            columns = ['predicted_streamflow']

        # Create a DataFrame for predictions
        # Note: X_tensor has length = total_len - lookback
        # So predictions start from common_dates[lookback:]
        lookback = self.preprocessor.lookback
        pred_df = pd.DataFrame(predictions, columns=columns, index=common_dates[lookback:])

        # Join predictions with the original averaged features
        result = features_avg.join(pred_df, how='outer')

        self.logger.info(f"Shape of final result: {result.shape}")
        self._log_memory_usage()
        return result

    def _save_model_checkpoint(self, path: Path):
        """Save the LSTM model and scalers to disk."""
        self.logger.info(f"Saving LSTM model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.preprocessor.feature_scaler,
            'target_scaler': self.preprocessor.target_scaler,
            'lookback': self.preprocessor.lookback,
            'output_size': self.preprocessor.output_size,
            'target_names': self.preprocessor.target_names
        }, path)
        self.logger.info("Model saved successfully")

    def _load_model_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load a LSTM model checkpoint from disk."""
        self.logger.info(f"Loading LSTM model from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
        return torch.load(path, map_location=self.device)

    def _log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
