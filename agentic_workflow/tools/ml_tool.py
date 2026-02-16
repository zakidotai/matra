"""
ML tool for density and generic property prediction workflows
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import UQpy.scientific_machine_learning as sml
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import shap
from typing import Optional

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


class BayesianRegressor(nn.Module):
    """Bayesian Neural Network Regressor using UQpy"""
    
    def __init__(self, in_features: int, hidden_layers: int, hidden_units: int, out_features: int = 1):
        super().__init__()
        layers = []
        last = in_features
        for _ in range(hidden_layers):
            layers.append(sml.BayesianLinear(last, hidden_units))
            layers.append(nn.ReLU())
            last = hidden_units
        layers.append(sml.BayesianLinear(last, out_features))
        self.network = nn.Sequential(*layers)
        self.wrapper = sml.FeedForwardNeuralNetwork(self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapper(x)

    def sample(self, flag: bool = True) -> None:
        self.wrapper.sample(flag)


def train_one_epoch(model: BayesianRegressor, loader: DataLoader, loss_fn, div_fn, optimizer, beta: float, device):
    """Train one epoch of Bayesian model"""
    model.train()
    model.sample(True)
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss = loss + beta * div_fn(model.wrapper)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def evaluate(model: BayesianRegressor, loader: DataLoader, loss_fn, device):
    """Evaluate Bayesian model"""
    model.eval()
    model.sample(False)
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def predict_with_uncertainty(model: BayesianRegressor, X: torch.Tensor, y_scaler: StandardScaler, n_samples: int = 200, device=None):
    """Predict with uncertainty using weight sampling"""
    if device is None:
        device = torch.device("cpu")
    
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            model.sample(True)
            p = model(X.to(device)).cpu().numpy().ravel()
            preds.append(p)
    preds = np.stack(preds, axis=0)
    # Inverse-transform from scaled target back to original units
    preds_unscaled = y_scaler.inverse_transform(preds)
    mean_pred = preds_unscaled.mean(axis=0)
    lower = np.percentile(preds_unscaled, 2.5, axis=0)
    upper = np.percentile(preds_unscaled, 97.5, axis=0)
    std_pred = preds_unscaled.std(axis=0)
    return mean_pred, lower, upper, std_pred


def clean_data(df: pd.DataFrame, target_column: str = "Density") -> pd.DataFrame:
    """
    Clean data following notebook logic - iterate until shape stabilizes
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        Cleaned dataframe
    """
    newdf = df.copy()
    # Drop index-like columns (e.g. Unnamed: 0) so they are not used as features or in sum filter
    index_like = [c for c in newdf.columns if str(c).startswith("Unnamed") or (isinstance(c, str) and c.strip() == "")]
    if index_like:
        newdf = newdf.drop(columns=index_like)
    prev_shape = None
    
    # Iterate cleaning until shape stabilizes
    max_iterations = 10
    for iteration in range(max_iterations):
        current_shape = newdf.shape
        
        # Filter rows where target > 0.01
        newdf = newdf[newdf[target_column] > 0.01]
        
        # Remove columns with < 10 non-zero values
        newdf = newdf.T[newdf.astype(bool).sum(axis=0) >= 10].T
        
        # Remove 'H' column if present
        cols = newdf.columns.tolist()
        if 'H' in cols:
            cols.remove('H')
        
        # Filter rows where feature sum (excluding target) is between 99-101
        if target_column in cols:
            feature_cols = [c for c in cols if c != target_column]
        else:
            feature_cols = cols[:-1]  # Assume last column is target
        
        if feature_cols:
            mask = newdf[feature_cols].sum(axis=1).between(99, 101)
            newdf = newdf[mask]
            
            # Remove columns with < 10 non-zero values again
            newdf = newdf.T[newdf.astype(bool).sum(axis=0) >= 10].T
        
        # Check if shape has stabilized
        if prev_shape == newdf.shape:
            logger.info(f"Data cleaning converged after {iteration + 1} iterations. Final shape: {newdf.shape}")
            break
        
        prev_shape = newdf.shape
    
    return newdf


def _safe_filename_suffix(name: str) -> str:
    """Return a filesystem-safe suffix from a property/column name."""
    s = re.sub(r"[^\w\-]", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() if s else "property"


# Common abbreviations for property columns (abbrev -> substrings that identify the column)
_TARGET_COLUMN_HINTS = {
    "ym": ["young", "modulus"],
    "e": ["young", "modulus"],
    "hv": ["hardness", "vickers"],
    "density": ["density"],
}


def _resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    """
    Resolve target_column to an actual column name in df.
    Tries exact match, case-insensitive match, substring match, then common abbreviations.
    """
    cols = list(df.columns)
    if not cols:
        raise KeyError(f"No columns in dataframe (target_column={target_column!r})")
    if target_column in cols:
        return target_column
    target_lower = target_column.strip().lower()
    for c in cols:
        if c.strip().lower() == target_lower:
            return c
    for c in cols:
        if target_lower in str(c).lower():
            return c
    # Try common abbreviations (e.g. YM -> column containing "young" and "modulus")
    hints = _TARGET_COLUMN_HINTS.get(target_lower)
    if hints:
        for c in cols:
            cl = str(c).lower()
            if all(h in cl for h in hints):
                return c
    raise KeyError(
        f"target_column {target_column!r} not found in dataframe. "
        f"Available columns: {cols}"
    )


def _run_ml_property_workflow(
    data_path: str,
    output_dir: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 94,
    n_trials: int = 100,
    n_samples_uncertainty: int = 300,
    hyperparams: Optional[dict] = None,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
) -> dict:
    """
    Internal: run full ML workflow for a given target property.
    Uses target_column for cleaning/labels and a filesystem-safe suffix for filenames.
    """
    suffix = _safe_filename_suffix(target_column)
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Data Loading
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        # Resolve target column (e.g. "YM" -> "Young's modulus")
        resolved_target = _resolve_target_column(df, target_column)
        if resolved_target != target_column:
            logger.info(f"Resolved target_column {target_column!r} -> {resolved_target!r}")
        
        # Step 1b: Histogram of property values (before filtering/cleaning) so user can set range
        vals = df[resolved_target].dropna()
        vals = vals[vals > 0.01]  # exclude zeros/negatives for display
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=min(50, max(10, len(vals) // 5)), edgecolor="gray", alpha=0.8)
            ax.set_xlabel(resolved_target)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {resolved_target} (before cleaning)\nmin={vals.min():.3g}, max={vals.max():.3g}, n={len(vals)}")
            ax.grid(True, alpha=0.3)
            histogram_path = os.path.join(output_dir, f"histogram_target_{suffix}.png")
            plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved target histogram to {histogram_path}")
        else:
            histogram_path = None
        
        # Step 1c: Apply optional target range filter (user can set target_min/target_max from histogram)
        if target_min is not None or target_max is not None:
            if target_min is not None:
                df = df[df[resolved_target] >= target_min]
                logger.info(f"Filtered to {resolved_target} >= {target_min}: {len(df)} rows")
            if target_max is not None:
                df = df[df[resolved_target] <= target_max]
                logger.info(f"Filtered to {resolved_target} <= {target_max}: {len(df)} rows")
        
        # Step 2: Data Cleaning
        logger.info("Cleaning data...")
        newdf = clean_data(df, resolved_target)
        logger.info(f"Cleaned data shape: {newdf.shape}")
        
        # Step 3: Prepare features and target
        x = newdf[newdf.columns[:-1]]  # All columns except last
        y = newdf[resolved_target]
        
        # Create family column (optional, for compatibility with notebook)
        family = []
        for i in range(len(newdf)):
            fam = ''.join(list(newdf.iloc[i][newdf.iloc[i].astype(bool)].index)[:-1])
            family.append(fam)
        x['family'] = family
        
        # Step 4: Train/Test Split
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        
        famtrain = X_train['family']
        famtest = X_test['family']
        
        # Get feature columns (excluding family)
        cols = [c for c in x.columns if c != 'family']
        X_train_features = X_train[cols]
        X_test_features = X_test[cols]
        
        # Step 5: Train Tree Model (Random Forest)
        logger.info("Training Random Forest model...")
        rf = RandomForestRegressor(n_estimators=60, random_state=random_state)
        rf.fit(X_train_features, y_train)
        
        # Evaluate tree model
        train_pred_rf = rf.predict(X_train_features)
        test_pred_rf = rf.predict(X_test_features)
        
        train_r2_rf = r2_score(y_train, train_pred_rf)
        test_r2_rf = r2_score(y_test, test_pred_rf)
        train_rmse_rf = np.sqrt(mean_squared_error(y_train, train_pred_rf))
        test_rmse_rf = np.sqrt(mean_squared_error(y_test, test_pred_rf))
        train_mae_rf = mean_absolute_error(y_train, train_pred_rf)
        test_mae_rf = mean_absolute_error(y_test, test_pred_rf)
        
        logger.info(f"Tree Model - Train R2: {train_r2_rf:.4f}, Test R2: {test_r2_rf:.4f}")
        
        # Step 6: Prepare data for Bayesian model
        device = torch.device("cpu")
        logger.info(f"Using {device} device")
        
        # Prepare numeric feature matrices without the 'family' column
        bn_cols = [c for c in cols]  # Already excludes family
        
        # Fit scalers on training data only
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train_features[bn_cols].values)
        X_test_scaled = X_scaler.transform(X_test_features[bn_cols].values)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
        
        # Torch tensors from scaled data
        X_train_bn = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_bn = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_bn = torch.tensor(y_train_scaled, dtype=torch.float32)
        y_test_bn = torch.tensor(y_test_scaled, dtype=torch.float32)
        
        # Datasets
        train_dataset = TensorDataset(X_train_bn, y_train_bn)
        test_dataset = TensorDataset(X_test_bn, y_test_bn)
        
        # Step 7: Optuna Hyperparameter Optimization
        logger.info(f"Running Optuna optimization with {n_trials} trials...")
        
        # Set default hyperparameter ranges if not provided
        if hyperparams is None:
            hyperparams = {}
        
        hidden_layers_min = hyperparams.get("hidden_layers_min", 2)
        hidden_layers_max = hyperparams.get("hidden_layers_max", 4)
        hidden_units_options = hyperparams.get("hidden_units_options", [32, 64, 128, 256, 512])
        lr_min = hyperparams.get("lr_min", 1e-4)
        lr_max = hyperparams.get("lr_max", 5e-2)
        weight_decay_min = hyperparams.get("weight_decay_min", 1e-8)
        weight_decay_max = hyperparams.get("weight_decay_max", 1e-2)
        beta_min = hyperparams.get("beta_min", 1e-8)
        beta_max = hyperparams.get("beta_max", 1e-4)
        batch_size_options = hyperparams.get("batch_size_options", [32, 64, 128])
        epochs_min = hyperparams.get("epochs_min", 80)
        epochs_max = hyperparams.get("epochs_max", 200)
        
        logger.info(f"Hyperparameter search space:")
        logger.info(f"  Hidden layers: {hidden_layers_min}-{hidden_layers_max}")
        logger.info(f"  Hidden units: {hidden_units_options}")
        logger.info(f"  Learning rate: {lr_min:.2e}-{lr_max:.2e}")
        logger.info(f"  Weight decay: {weight_decay_min:.2e}-{weight_decay_max:.2e}")
        logger.info(f"  Beta: {beta_min:.2e}-{beta_max:.2e}")
        logger.info(f"  Batch size: {batch_size_options}")
        logger.info(f"  Epochs: {epochs_min}-{epochs_max}")
        
        def create_loader(dataset, batch_size: int, shuffle: bool):
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        def objective(trial: optuna.Trial):
            hidden_layers = trial.suggest_int("hidden_layers", hidden_layers_min, hidden_layers_max)
            hidden_units = trial.suggest_categorical("hidden_units", hidden_units_options)
            lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
            weight_decay = trial.suggest_float("weight_decay", weight_decay_min, weight_decay_max, log=True)
            beta = trial.suggest_float("beta", beta_min, beta_max, log=True)
            batch_size = trial.suggest_categorical("batch_size", batch_size_options)
            epochs = trial.suggest_int("epochs", epochs_min, epochs_max)
            
            model = BayesianRegressor(in_features=len(bn_cols), hidden_layers=hidden_layers, hidden_units=hidden_units)
            model = model.to(device)
            
            train_loader = create_loader(train_dataset, batch_size, True)
            test_loader = create_loader(test_dataset, batch_size, False)
            
            loss_fn = nn.MSELoss()
            div_fn = sml.GaussianKullbackLeiblerDivergence(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            best_val = float("inf")
            no_improve = 0
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, train_loader, loss_fn, div_fn, optimizer, beta, device)
                val_loss = evaluate(model, test_loader, loss_fn, device)
                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 10:
                    break
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Best value (MSE): {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Step 8: Retrain best model
        logger.info("Retraining best model...")
        best_params = study.best_params
        hidden_layers = best_params["hidden_layers"]
        hidden_units = best_params["hidden_units"]
        lr = best_params["lr"]
        weight_decay = best_params["weight_decay"]
        beta = best_params["beta"]
        batch_size = best_params["batch_size"]
        epochs = best_params["epochs"]
        
        best_model = BayesianRegressor(in_features=len(bn_cols), hidden_layers=hidden_layers, hidden_units=hidden_units).to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        loss_fn = nn.MSELoss()
        div_fn = sml.GaussianKullbackLeiblerDivergence(device=device)
        optimizer = torch.optim.AdamW(best_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val = float("inf")
        no_improve = 0
        for epoch in range(epochs):
            train_loss = train_one_epoch(best_model, train_loader, loss_fn, div_fn, optimizer, beta, device)
            val_loss = evaluate(best_model, test_loader, loss_fn, device)
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 15:
                break
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")
        
        # Final metrics on train/test (inverse-transform predictions)
        best_model.sample(False)
        with torch.no_grad():
            train_pred_scaled = best_model(X_train_bn.to(device)).cpu().numpy().ravel()
            test_pred_scaled = best_model(X_test_bn.to(device)).cpu().numpy().ravel()
        
        train_pred_bn = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        test_pred_bn = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
        
        train_rmse_bn = float(np.sqrt(mean_squared_error(y_train.values, train_pred_bn)))
        test_rmse_bn = float(np.sqrt(mean_squared_error(y_test.values, test_pred_bn)))
        train_r2_bn = float(r2_score(y_train.values, train_pred_bn))
        test_r2_bn = float(r2_score(y_test.values, test_pred_bn))
        train_mae_bn = float(mean_absolute_error(y_train.values, train_pred_bn))
        test_mae_bn = float(mean_absolute_error(y_test.values, test_pred_bn))
        
        logger.info(f"Bayesian Model - Train R2: {train_r2_bn:.4f}, Test R2: {test_r2_bn:.4f}")
        
        # Step 9: Save models and scalers
        logger.info("Saving models and scalers...")
        model_path = os.path.join(output_dir, f"bayesian_model_{suffix}.pt")
        X_scaler_path = os.path.join(output_dir, f"X_scaler_{suffix}.pkl")
        y_scaler_path = os.path.join(output_dir, f"y_scaler_{suffix}.pkl")
        
        torch.save(best_model.state_dict(), model_path)
        joblib.dump(X_scaler, X_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)
        
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved scalers to {X_scaler_path} and {y_scaler_path}")
        
        # Step 10: Load and verify model
        logger.info("Verifying saved model...")
        loaded_model = BayesianRegressor(in_features=len(bn_cols), hidden_layers=hidden_layers, hidden_units=hidden_units).to(device)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_X_scaler = joblib.load(X_scaler_path)
        loaded_y_scaler = joblib.load(y_scaler_path)
        
        # Quick verification
        loaded_model.sample(False)
        with torch.no_grad():
            test_pred_loaded = loaded_model(X_test_bn.to(device)).cpu().numpy().ravel()
        test_pred_loaded_unscaled = loaded_y_scaler.inverse_transform(test_pred_loaded.reshape(-1, 1)).ravel()
        verification_r2 = r2_score(y_test.values, test_pred_loaded_unscaled)
        logger.info(f"Model verification R2: {verification_r2:.4f}")
        
        # Step 11: Generate visualizations
        logger.info("Generating visualizations...")
        
        # 11a. Scatter plot: actual vs predicted for train/test sets
        fig, ax = plt.subplots(figsize=(8, 6))
        mask_train = y_train.values > 1
        mask_test = y_test.values > 1
        ax.scatter(y_train.values[mask_train], train_pred_bn[mask_train], label='Train', alpha=0.6)
        ax.scatter(y_test.values[mask_test], test_pred_bn[mask_test], label='Test', alpha=0.6)
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', alpha=0.3)
        ax.set_xlabel(f'Actual {resolved_target}')
        ax.set_ylabel(f'Predicted {resolved_target}')
        ax.set_title(f'Bayesian Model: Actual vs Predicted ({resolved_target})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        scatter_plot_path = os.path.join(output_dir, f"scatter_actual_predicted_{suffix}.png")
        plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 11b. Scatter plot with error bars for test data
        mean_test, lower_test, upper_test, std_pred = predict_with_uncertainty(
            best_model, X_test_bn, y_scaler, n_samples=n_samples_uncertainty, device=device
        )
        results_test = pd.DataFrame({
            'mean_': mean_test,
            'lower': lower_test,
            'upper': upper_test,
            'actual': y_test.values,
            'std_': std_pred
        })
        results_test.index = y_test.index
        
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = results_test.actual > 1
        ax.errorbar(results_test.actual[mask], results_test.mean_[mask], 
                   yerr=results_test.std_[mask], fmt='o', capsize=5, alpha=0.6)
        ax.plot([results_test.actual.min(), results_test.actual.max()], 
               [results_test.actual.min(), results_test.actual.max()], 'r--', alpha=0.3)
        ax.set_xlabel(f'Actual {resolved_target}')
        ax.set_ylabel(f'Predicted {resolved_target}')
        ax.set_title(f'Bayesian Model: Test Data with Uncertainty ({resolved_target})')
        ax.grid(True, alpha=0.3)
        errorbar_plot_path = os.path.join(output_dir, f"scatter_errorbars_{suffix}.png")
        plt.savefig(errorbar_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 11c. SHAP for tree model
        logger.info("Computing SHAP values for tree model...")
        explainer_rf = shap.Explainer(rf)
        shap_values_rf = explainer_rf(X_train_features[bn_cols])
        
        # SHAP bar plot for tree model
        sv_rf_mean_abs = np.mean(np.abs(shap_values_rf.values), axis=0)
        dfsv_rf = pd.DataFrame(sv_rf_mean_abs).T
        dfsv_rf.columns = bn_cols
        dfsv_rf = dfsv_rf.T
        dfsv_rf = dfsv_rf.sort_values(by=0, ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(list(dfsv_rf.index)[::-1], list(dfsv_rf.values.ravel())[::-1], fc='cornflowerblue')
        ax.set_xlabel('mean |SHAP value|')
        ax.set_title('SHAP Feature Importance (Tree Model)')
        plt.tight_layout()
        shap_bar_rf_path = os.path.join(output_dir, f"shap_bar_tree_{suffix}.png")
        plt.savefig(shap_bar_rf_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # SHAP scatter plot for tree model
        fig = plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values_rf, max_display=30, show=False)
        plt.xlim(-2, 5)
        shap_scatter_rf_path = os.path.join(output_dir, f"shap_scatter_tree_{suffix}.png")
        plt.savefig(shap_scatter_rf_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 11d. SHAP for Bayesian model
        # SHAP values should be computed in the scaled space (where model was trained)
        # Then transformed appropriately for visualization with original feature values
        logger.info("Computing SHAP values for Bayesian model using DeepExplainer...")
        best_model.sample(False)
        best_model.eval()
        
        # Use scaled data for SHAP computation (same scale as training data)
        # Use entire training set for SHAP explanation (matching tree model approach)
        bg_size = min(100, len(X_train_scaled))
        background_scaled = X_train_bn[:bg_size]
        X_eval_scaled = X_train_bn  # Use entire training set like tree model
        
        # Compute SHAP values in scaled space (where the model operates)
        logger.info("Computing SHAP values for entire training set in scaled feature space...")
        explainer_bnn = shap.DeepExplainer(best_model, background_scaled)
        shap_values_raw = explainer_bnn.shap_values(X_eval_scaled)
        
        # Handle single-output regression return format
        if isinstance(shap_values_raw, list):
            shap_values_scaled = shap_values_raw[0]
        else:
            shap_values_scaled = shap_values_raw
        
        # Ensure numpy array
        if isinstance(shap_values_scaled, torch.Tensor):
            shap_values_scaled = shap_values_scaled.cpu().numpy()
        
        logger.info(f"SHAP values shape (before transformation): {shap_values_scaled.shape}")
        
        # Handle different SHAP output formats
        # For regression, SHAP values should be (n_samples, n_features) or (n_samples, n_features, n_outputs)
        # If 3D, we need to handle it appropriately
        if len(shap_values_scaled.shape) == 3:
            n_samples, dim1, dim2 = shap_values_scaled.shape
            if dim1 == dim2:
                # Shape is (n_samples, n_features, n_features) - might be interaction matrix
                # For SHAP, we want marginal contributions, so take diagonal elements
                logger.info(f"SHAP values have shape ({n_samples}, {dim1}, {dim2}), extracting diagonal contributions")
                shap_values_scaled = np.diagonal(shap_values_scaled, axis1=1, axis2=2)
            elif dim2 == 1:
                # Shape is (n_samples, n_features, 1) - single output, squeeze last dimension
                shap_values_scaled = shap_values_scaled.squeeze(axis=2)
            else:
                # Shape is (n_samples, n_features, n_outputs) - take first output
                logger.info(f"SHAP values have shape ({n_samples}, {dim1}, {dim2}), taking first output")
                shap_values_scaled = shap_values_scaled[:, :, 0]
        
        # Ensure 2D shape (n_samples, n_features)
        if len(shap_values_scaled.shape) > 2:
            shap_values_scaled = shap_values_scaled.squeeze()
        
        # Final check: ensure we have 2D array
        if len(shap_values_scaled.shape) != 2:
            raise ValueError(f"Expected 2D SHAP values, got shape {shap_values_scaled.shape}")
        
        logger.info(f"SHAP values shape (after shape handling): {shap_values_scaled.shape}")
        
        # SHAP values are computed in scaled space
        # According to SHAP docs: univariate input transformations (like StandardScaler) 
        # don't affect SHAP values - we just update the data attribute to show original features
        # However, since outputs are also scaled, we need to scale SHAP values by output_std
        # to convert contributions from scaled output space to original output space
        logger.info("Transforming SHAP values for output scaling...")
        
        # Get output scaling factor
        output_std = y_scaler.scale_[0]  # Standard deviation used for output scaling
        
        # Scale SHAP values by output_std to convert from scaled output to original output
        # Note: We DON'T transform by feature_stds because univariate input transformations
        # don't affect SHAP values (per SHAP documentation)
        shap_values_original = shap_values_scaled * output_std
        
        logger.info(f"SHAP values computed in scaled space, transformed for output scaling")
        logger.info(f"SHAP values shape (after transformation): {shap_values_original.shape}")
        
        # Use original feature values for visualization (univariate transformation doesn't affect SHAP values)
        # Use entire training set like tree model
        X_eval_original = X_train_features[bn_cols].values
        
        # SHAP bar plot for Bayesian model (transformed for output scaling)
        # Compute mean absolute SHAP values across samples
        sv_bnn_mean_abs = np.mean(np.abs(shap_values_original), axis=0)
        
        # Ensure sv_bnn_mean_abs is 1D
        if len(sv_bnn_mean_abs.shape) > 1:
            sv_bnn_mean_abs = sv_bnn_mean_abs.squeeze()
        
        shap_importance_df = pd.DataFrame({
            'feature': bn_cols,
            'mean_abs_shap': sv_bnn_mean_abs
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(shap_importance_df['feature'][::-1], shap_importance_df['mean_abs_shap'][::-1], color='cornflowerblue')
        ax.set_xlabel('mean |SHAP value| (original output space)')
        ax.set_title('SHAP Feature Importance (Bayesian Model)')
        plt.tight_layout()
        shap_bar_bnn_path = os.path.join(output_dir, f"shap_bar_bayesian_{suffix}.png")
        plt.savefig(shap_bar_bnn_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # SHAP scatter plot for Bayesian model (transformed for output scaling)
        # shap_values_original should already be 2D (n_samples, n_features) after earlier handling
        # Create a SHAP Explanation object with transformed values
        # For regression, base_values should be the mean prediction in original output space
        # Use already computed training predictions
        base_values = np.mean(train_pred_bn) if len(train_pred_bn) > 0 else 0.0
        
        shap_explanation = shap.Explanation(
            values=shap_values_original,
            base_values=np.full(len(X_eval_original), base_values),
            data=X_eval_original,  # Original feature values for context
            feature_names=bn_cols
        )
        
        fig = plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_explanation, max_display=30, show=False)
        shap_scatter_bnn_path = os.path.join(output_dir, f"shap_scatter_bayesian_{suffix}.png")
        plt.savefig(shap_scatter_bnn_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("All visualizations generated successfully")
        
        return {
            "success": True,
            "data_shape": newdf.shape,
            "model_paths": {
                "bayesian_model": model_path,
                "X_scaler": X_scaler_path,
                "y_scaler": y_scaler_path
            },
            "metrics": {
                "tree_model": {
                    "train_r2": float(train_r2_rf),
                    "test_r2": float(test_r2_rf),
                    "train_rmse": float(train_rmse_rf),
                    "test_rmse": float(test_rmse_rf),
                    "train_mae": float(train_mae_rf),
                    "test_mae": float(test_mae_rf)
                },
                "bayesian_model": {
                    "train_r2": float(train_r2_bn),
                    "test_r2": float(test_r2_bn),
                    "train_rmse": float(train_rmse_bn),
                    "test_rmse": float(test_rmse_bn),
                    "train_mae": float(train_mae_bn),
                    "test_mae": float(test_mae_bn),
                    "best_params": best_params
                }
            },
            "visualization_paths": {
                "histogram_target": histogram_path if histogram_path else None,
                "scatter_actual_predicted": scatter_plot_path,
                "scatter_errorbars": errorbar_plot_path,
                "shap_bar_tree": shap_bar_rf_path,
                "shap_scatter_tree": shap_scatter_rf_path,
                "shap_bar_bayesian": shap_bar_bnn_path,
                "shap_scatter_bayesian": shap_scatter_bnn_path
            },
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error in ML workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "data_shape": None,
            "model_paths": {},
            "metrics": {},
            "visualization_paths": {},
            "error": error_msg
        }


def ml_density_prediction_tool(
    data_path: str,
    output_dir: str,
    target_column: str = "Density",
    test_size: float = 0.2,
    random_state: int = 94,
    n_trials: int = 100,
    n_samples_uncertainty: int = 300,
    hyperparams: Optional[dict] = None,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
) -> dict:
    """
    Perform complete ML workflow for density prediction (same as property prediction with target_column='Density').
    """
    return _run_ml_property_workflow(
        data_path=data_path,
        output_dir=output_dir,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        n_trials=n_trials,
        n_samples_uncertainty=n_samples_uncertainty,
        hyperparams=hyperparams,
        target_min=target_min,
        target_max=target_max,
    )


def ml_property_prediction_tool(
    data_path: str,
    output_dir: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 94,
    n_trials: int = 100,
    n_samples_uncertainty: int = 300,
    hyperparams: Optional[dict] = None,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
) -> dict:
    """
    Perform complete ML workflow to predict any numeric property.

    Uses the same pipeline as density prediction: data cleaning, Random Forest + Bayesian
    model with Optuna tuning, and SHAP explainability. Specify the target property via
    target_column (e.g. 'Density', 'Youngs_Modulus', 'Hardness'). Output files use a
    filesystem-safe suffix derived from target_column. A histogram of the target is saved
    before cleaning so you can set target_min/target_max to restrict the value range.
    """
    return _run_ml_property_workflow(
        data_path=data_path,
        output_dir=output_dir,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        n_trials=n_trials,
        n_samples_uncertainty=n_samples_uncertainty,
        hyperparams=hyperparams,
        target_min=target_min,
        target_max=target_max,
    )


# Register the tool
registry = get_registry()
registry.register(
    name="ml_density_prediction",
    func=ml_density_prediction_tool,
    description="Perform ML workflow for density prediction: load data, save histogram of target (for setting range), optional target_min/target_max filter, then data cleaning, tree + Bayesian model training with Optuna, and SHAP explainability",
    parameters={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to CSV data file"
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for saving models and outputs"
            },
            "target_column": {
                "type": "string",
                "description": "Name of target column",
                "default": "Density"
            },
            "test_size": {
                "type": "number",
                "description": "Train/test split ratio",
                "default": 0.2
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed for reproducibility",
                "default": 94
            },
            "n_trials": {
                "type": "integer",
                "description": "Number of Optuna trials for hyperparameter optimization",
                "default": 100
            },
            "n_samples_uncertainty": {
                "type": "integer",
                "description": "Number of samples for Bayesian uncertainty estimation",
                "default": 300
            },
            "target_min": {
                "type": "number",
                "description": "Optional minimum value of target to include (set after inspecting histogram_target plot)"
            },
            "target_max": {
                "type": "number",
                "description": "Optional maximum value of target to include (set after inspecting histogram_target plot)"
            },
            "hyperparams": {
                "type": "object",
                "description": "Dictionary with hyperparameter search space configuration (optional)",
                "properties": {
                    "hidden_layers_min": {"type": "integer"},
                    "hidden_layers_max": {"type": "integer"},
                    "hidden_units_options": {"type": "array", "items": {"type": "integer"}},
                    "lr_min": {"type": "number"},
                    "lr_max": {"type": "number"},
                    "weight_decay_min": {"type": "number"},
                    "weight_decay_max": {"type": "number"},
                    "beta_min": {"type": "number"},
                    "beta_max": {"type": "number"},
                    "batch_size_options": {"type": "array", "items": {"type": "integer"}},
                    "epochs_min": {"type": "integer"},
                    "epochs_max": {"type": "integer"}
                }
            }
        },
        "required": ["data_path", "output_dir"]
    }
)

registry.register(
    name="ml_property_prediction",
    func=ml_property_prediction_tool,
    description="Perform ML workflow to predict any numeric property: load data, save histogram of target (for setting range), optional target_min/target_max filter, then data cleaning, tree + Bayesian model training with Optuna, and SHAP explainability. Specify target column (e.g. Density, YM, Hardness) via target_column.",
    parameters={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to CSV data file"
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for saving models and outputs"
            },
            "target_column": {
                "type": "string",
                "description": "Name of the target property column to predict (e.g. Density, Youngs_Modulus, Hardness)"
            },
            "test_size": {
                "type": "number",
                "description": "Train/test split ratio",
                "default": 0.2
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed for reproducibility",
                "default": 94
            },
            "n_trials": {
                "type": "integer",
                "description": "Number of Optuna trials for hyperparameter optimization",
                "default": 100
            },
            "n_samples_uncertainty": {
                "type": "integer",
                "description": "Number of samples for Bayesian uncertainty estimation",
                "default": 300
            },
            "target_min": {
                "type": "number",
                "description": "Optional minimum value of target to include (set after inspecting histogram_target plot)"
            },
            "target_max": {
                "type": "number",
                "description": "Optional maximum value of target to include (set after inspecting histogram_target plot)"
            },
            "hyperparams": {
                "type": "object",
                "description": "Dictionary with hyperparameter search space configuration (optional)",
                "properties": {
                    "hidden_layers_min": {"type": "integer"},
                    "hidden_layers_max": {"type": "integer"},
                    "hidden_units_options": {"type": "array", "items": {"type": "integer"}},
                    "lr_min": {"type": "number"},
                    "lr_max": {"type": "number"},
                    "weight_decay_min": {"type": "number"},
                    "weight_decay_max": {"type": "number"},
                    "beta_min": {"type": "number"},
                    "beta_max": {"type": "number"},
                    "batch_size_options": {"type": "array", "items": {"type": "integer"}},
                    "epochs_min": {"type": "integer"},
                    "epochs_max": {"type": "integer"}
                }
            }
        },
        "required": ["data_path", "output_dir", "target_column"]
    }
)

