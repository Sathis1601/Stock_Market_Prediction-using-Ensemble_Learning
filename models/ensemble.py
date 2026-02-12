"""
Ensemble Module for Stock Market Volatility Prediction
Combines predictions from LSTM, GRU, and CNN models
Implements uncertainty quantification through model disagreement
"""

import numpy as np
import yaml
from pathlib import Path
from scipy import stats


class EnsembleVolatilityPredictor:
    """
    Ensemble predictor combining LSTM, GRU, and CNN models
    Provides uncertainty estimation through prediction variance
    """
    
    def __init__(self, lstm_model, gru_model, cnn_model, config_path='config.yaml'):
        """
        Initialize ensemble with trained models
        
        Parameters:
        -----------
        lstm_model : LSTMVolatilityModel
            Trained LSTM model
        gru_model : GRUVolatilityModel
            Trained GRU model
        cnn_model : CNNVolatilityModel
            Trained CNN model
        """
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.cnn_model = cnn_model
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ensemble_config = self.config['ensemble']
        self.risk_thresholds = self.config['risk_thresholds']
        
        # Model weights (can be learned or predefined)
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for weighted averaging"""
        weights = self.ensemble_config.get('weights')
        
        if weights is None:
            # Equal weights by default
            weights = [1/3, 1/3, 1/3]
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def predict(self, X):
        """
        Make ensemble predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (samples, timesteps, features)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - ensemble_prediction: Final ensemble prediction
            - individual_predictions: Predictions from each model
            - uncertainty: Uncertainty measure (std/variance)
            - confidence: Confidence score (inverse of uncertainty)
        """
        # Get predictions from each model
        lstm_pred = self.lstm_model.predict(X)
        gru_pred = self.gru_model.predict(X)
        cnn_pred = self.cnn_model.predict(X)
        
        # Stack predictions
        all_predictions = np.stack([lstm_pred, gru_pred, cnn_pred], axis=1)
        
        # Calculate ensemble prediction based on method
        method = self.ensemble_config['method']
        
        if method == 'simple_average':
            ensemble_pred = np.mean(all_predictions, axis=1)
        elif method == 'weighted_average':
            ensemble_pred = np.average(all_predictions, axis=1, weights=self.weights)
        elif method == 'voting':
            # For regression, use median as voting
            ensemble_pred = np.median(all_predictions, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Calculate uncertainty
        uncertainty_method = self.ensemble_config['uncertainty_method']
        
        if uncertainty_method == 'variance':
            uncertainty = np.var(all_predictions, axis=1)
        elif uncertainty_method == 'std':
            uncertainty = np.std(all_predictions, axis=1)
        elif uncertainty_method == 'entropy':
            # For regression, use coefficient of variation as proxy
            uncertainty = np.std(all_predictions, axis=1) / (np.mean(all_predictions, axis=1) + 1e-8)
        else:
            uncertainty = np.std(all_predictions, axis=1)
        
        # Confidence score based on a fixed reference scale
        # Typical volatility uncertainty ranges from 0 to 0.05 (5%)
        # Lower uncertainty = higher confidence
        reference_uncertainty = 0.05  # Maximum expected uncertainty (5% volatility)
        confidence = np.clip(1 - (uncertainty / reference_uncertainty), 0, 1)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': {
                'lstm': lstm_pred,
                'gru': gru_pred,
                'cnn': cnn_pred
            },
            'uncertainty': uncertainty,
            'confidence': confidence
        }
    
    def classify_risk(self, volatility):
        """
        Classify risk level based on predicted volatility
        
        Parameters:
        -----------
        volatility : float or np.ndarray
            Predicted volatility value(s)
            
        Returns:
        --------
        str or np.ndarray
            Risk classification: 'Low', 'Medium', or 'High'
        """
        thresholds = self.risk_thresholds
        
        def classify_single(vol):
            if vol < thresholds['low']:
                return 'Low'
            elif vol < thresholds['medium']:
                return 'Medium'
            else:
                return 'High'
        
        if isinstance(volatility, (int, float)):
            return classify_single(volatility)
        else:
            return np.array([classify_single(v) for v in volatility])
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate ensemble performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
            
        Returns:
        --------
        dict
            Comprehensive evaluation metrics
        """
        # Get predictions
        results = self.predict(X_test)
        y_pred = results['ensemble_prediction']
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0
        
        # Mean uncertainty
        mean_uncertainty = np.mean(results['uncertainty'])
        mean_confidence = np.mean(results['confidence'])
        
        # Individual model metrics
        individual_metrics = {}
        for model_name, pred in results['individual_predictions'].items():
            model_mse = np.mean((y_test - pred) ** 2)
            model_mae = np.mean(np.abs(y_test - pred))
            individual_metrics[model_name] = {
                'mse': model_mse,
                'mae': model_mae,
                'rmse': np.sqrt(model_mse)
            }
        
        metrics = {
            'ensemble': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mean_uncertainty': mean_uncertainty,
                'mean_confidence': mean_confidence
            },
            'individual_models': individual_metrics
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nEnsemble Performance:")
        print(f"  MSE:                    {mse:.8f}")
        print(f"  MAE:                    {mae:.8f}")
        print(f"  RMSE:                   {rmse:.8f}")
        print(f"  R²:                     {r2:.4f}")
        print(f"  Directional Accuracy:   {directional_accuracy:.4f}")
        print(f"  Mean Uncertainty:       {mean_uncertainty:.8f}")
        print(f"  Mean Confidence:        {mean_confidence:.4f}")
        
        print(f"\nIndividual Model Performance:")
        for model_name, metrics in individual_metrics.items():
            print(f"  {model_name.upper()}:")
            print(f"    MSE:  {metrics['mse']:.8f}")
            print(f"    MAE:  {metrics['mae']:.8f}")
            print(f"    RMSE: {metrics['rmse']:.8f}")
        
        # Calculate improvement
        ensemble_rmse = rmse
        avg_individual_rmse = np.mean([m['rmse'] for m in individual_metrics.values()])
        improvement = ((avg_individual_rmse - ensemble_rmse) / avg_individual_rmse) * 100
        
        print(f"\nEnsemble Improvement:")
        print(f"  Average Individual RMSE: {avg_individual_rmse:.8f}")
        print(f"  Ensemble RMSE:           {ensemble_rmse:.8f}")
        print(f"  Improvement:             {improvement:.2f}%")
        
        print("=" * 60)
        
        return metrics
    
    def update_weights(self, X_val, y_val):
        """
        Update ensemble weights based on validation performance
        
        Parameters:
        -----------
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation targets
        """
        # Get predictions from each model
        lstm_pred = self.lstm_model.predict(X_val)
        gru_pred = self.gru_model.predict(X_val)
        cnn_pred = self.cnn_model.predict(X_val)
        
        # Calculate MSE for each model
        lstm_mse = np.mean((y_val - lstm_pred) ** 2)
        gru_mse = np.mean((y_val - gru_pred) ** 2)
        cnn_mse = np.mean((y_val - cnn_pred) ** 2)
        
        # Inverse MSE weighting (better models get higher weights)
        inverse_mse = np.array([1/lstm_mse, 1/gru_mse, 1/cnn_mse])
        weights = inverse_mse / inverse_mse.sum()
        
        self.weights = weights
        
        print(f"\nUpdated ensemble weights:")
        print(f"  LSTM: {weights[0]:.4f}")
        print(f"  GRU:  {weights[1]:.4f}")
        print(f"  CNN:  {weights[2]:.4f}")
    
    def get_prediction_intervals(self, X, confidence_level=0.95):
        """
        Calculate prediction intervals using ensemble uncertainty
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        confidence_level : float
            Confidence level for intervals (default: 0.95)
            
        Returns:
        --------
        dict
            Prediction with intervals
        """
        results = self.predict(X)
        ensemble_pred = results['ensemble_prediction']
        uncertainty = results['uncertainty']
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Prediction intervals
        lower_bound = ensemble_pred - z_score * uncertainty
        upper_bound = ensemble_pred + z_score * uncertainty
        
        return {
            'prediction': ensemble_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }


if __name__ == "__main__":
    print("Ensemble Volatility Prediction Module")
    print("This module combines LSTM, GRU, and CNN predictions with uncertainty quantification.")
