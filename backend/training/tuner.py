import optuna
import torch
import torch.nn as nn
from backend.models.lstm_attention import LSTMAttentionModel
from backend.core.logging import logger
from typing import Dict, Any, Tuple

class HyperparameterTuner:
    """
    Optimizes LSTM hyperparameters using Optuna.
    """
    def __init__(self, train_loader, val_loader, input_dim):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim

    def objective(self, trial) -> float:
        """
        Optuna objective function.
        """
        # Define search space
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        
        # Initialize model
        model = LSTMAttentionModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Quick training loop (e.g., 5 epochs for tuning)
        model.train()
        for epoch in range(5):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val)
                    val_loss += criterion(val_outputs, y_val).item()
            
            # Pruning (stop unpromising trials early)
            avg_val_loss = val_loss / len(self.val_loader)
            trial.report(avg_val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            model.train()
            
        return avg_val_loss

    def optimize(self, n_trials=10) -> Dict[str, Any]:
        """
        Run optimization.
        """
        logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params
