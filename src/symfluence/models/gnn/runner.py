
"""
GNN Model Runner.

Orchestrates the GNN model workflow: data loading, graph construction, training, and simulation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ..registry import ModelRegistry
from ..base import BaseModelRunner
from ..execution import ModelExecutor, SpatialOrchestrator
from symfluence.core.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)

from .model import GNNModel
from .preprocessor import GNNPreprocessor
from .postprocessor import GNNPostprocessor

@ModelRegistry.register_runner('GNN', method_name='run_gnn')
class GNNRunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator):
    """
    Runner for the Spatio-Temporal GNN Hydrological Model.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized GNN runner with device: {self.device}")
        
        # Check spatial mode
        domain_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        if domain_method == 'lumped':
            self.logger.warning(
                "⚠️  GNN model requested in 'lumped' mode. GNN is designed for spatially distributed modeling "
                "with a graph structure. Consider using 'LSTM' for lumped modeling or change "
                "DOMAIN_DEFINITION_METHOD to 'delineate'."
            )

        self.preprocessor = GNNPreprocessor(
            self.config_dict, 
            self.logger, 
            self.project_dir, 
            self.device
        )
        self.postprocessor = GNNPostprocessor(
            self.config_dict,
            self.logger,
            reporting_manager=self.reporting_manager
        )
        
        self.model = None
        self.hru_ids = []
        self.outlet_indices = []
        self.outlet_hru_ids = []

    def _get_model_name(self) -> str:
        return "GNN"

    def run_gnn(self):
        """Run the complete GNN model workflow."""
        self.logger.info("Starting GNN model run")

        with symfluence_error_handler(
            "GNN model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # 1. Load Data & Graph
            forcing_df, streamflow_df, snow_df = self.preprocessor.load_data()
            
            # Load Graph to get adjacency
            adj_matrix = self.preprocessor.load_graph_structure()
            
            # 2. Preprocess
            use_snow = self.config_dict.get('GNN_USE_SNOW', False)
            snow_df_input = snow_df if use_snow else pd.DataFrame()

            load_existing_model = self.config_dict.get('GNN_LOAD', False)
            model_save_path = self.project_dir / 'models' / 'gnn_model.pt'
            
            if load_existing_model:
                self.logger.info("Loading pre-trained GNN model")
                checkpoint = self._load_model_checkpoint(model_save_path)
                
                self.preprocessor.set_scalers(
                    checkpoint['feature_scaler'], 
                    checkpoint['target_scaler'],
                    checkpoint['output_size'],
                    checkpoint['target_names']
                )
                
                # Check if graph matches
                if checkpoint['adj_matrix_shape'] != list(adj_matrix.shape):
                    self.logger.warning("Loaded model graph shape mismatch! This may cause errors.")
                
                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=False
                )
                self.hru_ids = hru_ids
                self.outlet_indices = self.preprocessor.outlet_indices
                self.outlet_hru_ids = self.preprocessor.outlet_hru_ids
                
                self._create_model_instance(
                    input_size=X_tensor.shape[-1],
                    hidden_size=self.config_dict.get('GNN_HIDDEN_SIZE', 64),
                    gnn_output_size=self.config_dict.get('GNN_OUTPUT_SIZE', 32),
                    adjacency_matrix=adj_matrix
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            else:
                # Train
                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=True
                )
                self.hru_ids = hru_ids
                self.outlet_indices = self.preprocessor.outlet_indices
                self.outlet_hru_ids = self.preprocessor.outlet_hru_ids
                
                self._create_model_instance(
                    input_size=X_tensor.shape[-1],
                    hidden_size=self.config_dict.get('GNN_HIDDEN_SIZE', 64),
                    gnn_output_size=self.config_dict.get('GNN_OUTPUT_SIZE', 32),
                    adjacency_matrix=adj_matrix
                )
                
                self._train_model(
                    X_tensor, 
                    y_tensor,
                    epochs=self.config_dict.get('GNN_EPOCHS', 100),
                    batch_size=self.config_dict.get('GNN_BATCH_SIZE', 16), # Smaller batch due to graph size
                    learning_rate=self.config_dict.get('GNN_LEARNING_RATE', 0.005)
                )
                
                self.project_dir.joinpath('models').mkdir(exist_ok=True)
                self._save_model_checkpoint(model_save_path, adj_matrix)

            # 3. Simulate
            results = self._simulate(X_tensor, common_dates, hru_ids)
            
            # 4. Save Results
            output_file = self.postprocessor.save_results(
                results,
                hru_ids=self.hru_ids,
                outlet_hru_ids=self.outlet_hru_ids
            )
            self.logger.info(f"Results saved to {output_file}")

    def _create_model_instance(self, input_size, hidden_size, gnn_output_size, adjacency_matrix):
        """Create the GNN model instance."""
        self.logger.info(f"Creating GNN model: In={input_size}, Hidden={hidden_size}, GNN_Out={gnn_output_size}")
        self.model = GNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            gnn_output_size=gnn_output_size,
            adjacency_matrix=adjacency_matrix,
            dropout_rate=float(self.config_dict.get('GNN_DROPOUT', 0.2))
        ).to(self.device)

    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the GNN model.
        X: (B, T, N, F)
        y: (B, N, O) - Target streamflow at outlets (others may be 0/masked)
        """
        self.logger.info(f"Training GNN with {epochs} epochs, batch_size: {batch_size}")
        
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Mask: Identify which nodes have valid targets.
        # Assuming we only have data at outlets, and non-outlets are 0.
        # But scaling might have shifted 0. 
        # A robust way is to check the variance of targets or use explicit mask.
        # Here we assume any node with non-constant target is observed?
        # Simpler: In preprocessor, we set outlets.
        # Let's derive mask from y: if y is constant (0-like) across time, maybe unobserved?
        # Actually, streamflow varies. 
        # We will assume only the Outlet nodes contribute to loss.
        # To do this generically, we compute loss on all nodes but weight them?
        # Or just compute loss on nodes where we have data.
        # Since we don't pass a mask explicitly, we will assume y contains valid data where it matters.
        # If internal nodes are 0, the model will learn to predict 0 there? That's BAD.
        
        # FIX: We need a mask.
        # Let's create a mask based on variance of y in training set?
        # Or just use the 'outlets' logic again.
        # For now, let's assume y is correctly populated for outlets and we want to fit those.
        # If y is 0 for internal nodes, minimizing MSE will force flow to 0, which is physically wrong (water exists).
        # We should MASK out internal nodes from the loss.
        
        # Heuristic: Nodes with sum(abs(y)) > epsilon are observed.
        # Or pass mask from preprocessor.
        # Let's compute a mask on the whole dataset once.
        # y is (B, N, O). Sum over B and O.
        if self.outlet_indices:
            mask = torch.zeros(y.size(1), device=self.device)
            mask[self.outlet_indices] = 1.0
        else:
            y_activity = y.abs().sum(dim=(0, 2))
            mask = (y_activity > 1e-6).float().to(self.device) # (N,)
        self.logger.info(f"Training mask active for {mask.sum().item()} nodes out of {len(mask)}")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction='none') # We will apply mask
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle batches
            indices = torch.randperm(X_train.size(0))
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X) # (B, N, 1)
                
                # Reshape batch_y if needed (B, N, O) -> (B, N, 1) for streamflow
                target = batch_y[:, :, 0:1] # Take streamflow
                
                loss_raw = criterion(outputs, target) # (B, N, 1)
                
                # Apply mask: (B, N, 1) * (N, 1 broadcast)
                masked_loss = loss_raw * mask.view(1, -1, 1)
                
                loss = masked_loss.sum() / (mask.sum() * batch_X.size(0) + 1e-6)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_target = y_val[:, :, 0:1]
                val_loss_raw = criterion(val_out, val_target)
                val_loss = (val_loss_raw * mask.view(1, -1, 1)).sum() / (mask.sum() * X_val.size(0) + 1e-6)
            
            if (epoch + 1) % 10 == 0:
                avg_train_loss = total_loss / max(total_samples, 1)
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

    def _simulate(self, X: torch.Tensor, common_dates: pd.DatetimeIndex, hru_ids: List[int]) -> pd.DataFrame:
        """Run full simulation."""
        self.logger.info("Running GNN simulation")
        self.model.eval()
        
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(batch[0]) # (B, N, 1)
                predictions.append(out.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0) # (Total_Steps, N, 1)
        
        # Inverse transform
        # Target scaler was fitted on (Total, 1) or (Total*N, 1)
        # We need to apply inverse transform.
        # Our target scaler expects shape (..., 1) usually.
        B, N, O = predictions.shape
        preds_flat = predictions.reshape(-1, 1)
        preds_unscaled = self.preprocessor.target_scaler.inverse_transform(preds_flat)
        preds_restored = preds_unscaled.reshape(B, N, O)
        
        # Create DataFrame
        # MultiIndex: (Time, HRU)
        lookback = self.preprocessor.lookback
        time_idx = common_dates[lookback:]
        
        dfs = []
        for i, hru_id in enumerate(hru_ids):
            # Extract time series for this HRU
            data = preds_restored[:, i, 0]
            df = pd.DataFrame({'predicted_streamflow': data}, index=time_idx)
            df['hruId'] = hru_id
            dfs.append(df)
            
        result = pd.concat(dfs).reset_index().rename(columns={'index': 'time'})
        result = result.set_index(['time', 'hruId']).sort_index()
        
        return result

    def _save_model_checkpoint(self, path: Path, adj_matrix: torch.Tensor):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.preprocessor.feature_scaler,
            'target_scaler': self.preprocessor.target_scaler,
            'lookback': self.preprocessor.lookback,
            'output_size': self.preprocessor.output_size,
            'target_names': self.preprocessor.target_names,
            'adj_matrix_shape': list(adj_matrix.shape)
        }, path)

    def _load_model_checkpoint(self, path: Path):
        return torch.load(path, map_location=self.device)
