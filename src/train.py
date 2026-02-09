"""
Training Pipeline for Ensemble Volatility Prediction
Orchestrates data preparation, model training, and evaluation
"""

import sys
import os
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import StockDataCollector
from src.preprocessing import FeatureEngineer
from models.lstm_model import LSTMVolatilityModel
from models.gru_model import GRUVolatilityModel
from models.cnn_model import CNNVolatilityModel
from models.ensemble import EnsembleVolatilityPredictor


class TrainingPipeline:
    """
    Complete training pipeline for ensemble volatility prediction
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize training pipeline"""
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.paths_config = self.config['paths']
        
        # Create directories
        Path(self.paths_config['models']).mkdir(parents=True, exist_ok=True)
        Path(self.paths_config['logs']).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = StockDataCollector(config_path)
        self.engineer = FeatureEngineer(config_path)
        
        self.lstm_model = None
        self.gru_model = None
        self.cnn_model = None
        self.ensemble = None
    
    def prepare_data(self, ticker=None):
        """
        Prepare data for training
        
        Parameters:
        -----------
        ticker : str, optional
            Stock ticker (default: from config)
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if ticker is None:
            ticker = self.config['data']['default_ticker']
        
        print("\n" + "=" * 60)
        print(f"PREPARING DATA FOR {ticker}")
        print("=" * 60)
        
        # Step 1: Fetch data
        print("\n[1/5] Fetching stock data...")
        df = self.collector.fetch_stock_data(ticker)
        
        # Step 2: Calculate features
        print("\n[2/5] Calculating technical indicators...")
        df_processed = self.engineer.calculate_all_features(df)
        df_processed = self.engineer.prepare_target_variable(df_processed)
        
        # Step 3: Create sequences
        print("\n[3/5] Creating time-series sequences...")
        feature_columns = self.engineer.get_feature_columns()
        X, y = self.engineer.create_sequences(df_processed, feature_columns)
        
        # Step 4: Split data
        print("\n[4/5] Splitting data into train/val/test sets...")
        test_split = self.training_config['test_split']
        val_split = self.training_config['validation_split']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split / (1 - test_split), shuffle=False
        )
        
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Val set:   {X_val.shape[0]} samples")
        print(f"  Test set:  {X_test.shape[0]} samples")
        
        # Step 5: Normalize features
        print("\n[5/5] Normalizing features...")
        X_train, X_val, X_test = self.engineer.normalize_features(
            X_train, X_val, X_test
        )
        
        # Save scaler
        scaler_path = Path(self.paths_config['models']) / 'scaler.pkl'
        self.engineer.save_scaler(scaler_path)
        
        print("\n✓ Data preparation complete!")
        print("=" * 60)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """
        Train all three models independently
        
        Parameters:
        -----------
        X_train, X_val : np.ndarray
            Training and validation features
        y_train, y_val : np.ndarray
            Training and validation targets
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        print("\n" + "=" * 60)
        print("TRAINING ENSEMBLE MODELS")
        print("=" * 60)
        
        # Train LSTM
        print("\n" + "-" * 60)
        print("Training LSTM Model (1/3)")
        print("-" * 60)
        self.lstm_model = LSTMVolatilityModel(self.config_path)
        self.lstm_model.build_model(input_shape)
        self.lstm_model.train(X_train, y_train, X_val, y_val)
        
        # Save LSTM model
        lstm_path = Path(self.paths_config['models']) / 'lstm_model.keras'
        self.lstm_model.save_model(lstm_path)
        
        # Train GRU
        print("\n" + "-" * 60)
        print("Training GRU Model (2/3)")
        print("-" * 60)
        self.gru_model = GRUVolatilityModel(self.config_path)
        self.gru_model.build_model(input_shape)
        self.gru_model.train(X_train, y_train, X_val, y_val)
        
        # Save GRU model
        gru_path = Path(self.paths_config['models']) / 'gru_model.keras'
        self.gru_model.save_model(gru_path)
        
        # Train CNN
        print("\n" + "-" * 60)
        print("Training 1D-CNN Model (3/3)")
        print("-" * 60)
        self.cnn_model = CNNVolatilityModel(self.config_path)
        self.cnn_model.build_model(input_shape)
        self.cnn_model.train(X_train, y_train, X_val, y_val)
        
        # Save CNN model
        cnn_path = Path(self.paths_config['models']) / 'cnn_model.keras'
        self.cnn_model.save_model(cnn_path)
        
        print("\n✓ All models trained successfully!")
        print("=" * 60)
    
    def create_ensemble(self, X_val, y_val):
        """
        Create ensemble and optimize weights
        
        Parameters:
        -----------
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation targets
        """
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE")
        print("=" * 60)
        
        # Create ensemble
        self.ensemble = EnsembleVolatilityPredictor(
            self.lstm_model,
            self.gru_model,
            self.cnn_model,
            self.config_path
        )
        
        # Optimize weights based on validation performance
        print("\nOptimizing ensemble weights...")
        self.ensemble.update_weights(X_val, y_val)
        
        print("\n✓ Ensemble created successfully!")
        print("=" * 60)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate individual models and ensemble
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODELS")
        print("=" * 60)
        
        # Evaluate individual models
        print("\nIndividual Model Performance:")
        print("-" * 60)
        
        lstm_metrics = self.lstm_model.evaluate(X_test, y_test)
        gru_metrics = self.gru_model.evaluate(X_test, y_test)
        cnn_metrics = self.cnn_model.evaluate(X_test, y_test)
        
        # Evaluate ensemble
        print("-" * 60)
        ensemble_metrics = self.ensemble.evaluate_ensemble(X_test, y_test)
        
        return {
            'lstm': lstm_metrics,
            'gru': gru_metrics,
            'cnn': cnn_metrics,
            'ensemble': ensemble_metrics
        }
    
    def run_full_pipeline(self, ticker=None):
        """
        Run the complete training pipeline
        
        Parameters:
        -----------
        ticker : str, optional
            Stock ticker to train on
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "ENSEMBLE VOLATILITY PREDICTION TRAINING")
        print("=" * 80)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(ticker)
        
        # Train models
        self.train_models(X_train, X_val, y_train, y_val)
        
        # Create ensemble
        self.create_ensemble(X_val, y_val)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        print("\n" + "=" * 80)
        print(" " * 25 + "TRAINING COMPLETE!")
        print("=" * 80)
        
        return metrics


def main():
    """Run the training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ensemble volatility prediction models')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Stock ticker symbol (default: from config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(args.config)
    metrics = pipeline.run_full_pipeline(args.ticker)
    
    print("\nTraining pipeline completed successfully!")
    print(f"Models saved to: {pipeline.paths_config['models']}")


if __name__ == "__main__":
    main()
