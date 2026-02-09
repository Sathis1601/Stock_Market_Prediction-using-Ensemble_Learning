"""
GRU Model for Stock Market Volatility Prediction
Captures short-term market movements with faster computation than LSTM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import yaml


class GRUVolatilityModel:
    """
    GRU-based model for predicting stock market volatility
    Designed to capture short-term market movements efficiently
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize GRU model with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['gru']
        self.training_config = self.config['training']
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        Build GRU architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (timesteps, features)
            
        Returns:
        --------
        keras.Model
            Compiled GRU model
        """
        # Extract configuration
        units = self.model_config['units']
        dropout = self.model_config['dropout']
        activation = self.model_config['activation']
        learning_rate = self.training_config['learning_rate']
        
        # Build model
        model = models.Sequential(name='GRU_Volatility_Model')
        
        # First GRU layer (return sequences for stacking)
        model.add(layers.GRU(
            units[0],
            activation=activation,
            return_sequences=True,
            input_shape=input_shape,
            name='gru_layer_1'
        ))
        model.add(layers.Dropout(dropout, name='dropout_1'))
        
        # Second GRU layer
        model.add(layers.GRU(
            units[1],
            activation=activation,
            return_sequences=True,
            name='gru_layer_2'
        ))
        model.add(layers.Dropout(dropout, name='dropout_2'))
        
        # Third GRU layer (no return sequences)
        model.add(layers.GRU(
            units[2],
            activation=activation,
            return_sequences=False,
            name='gru_layer_3'
        ))
        model.add(layers.Dropout(dropout, name='dropout_3'))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout / 2, name='dropout_4'))
        
        model.add(layers.Dense(16, activation='relu', name='dense_2'))
        
        # Output layer (single value: volatility prediction)
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.training_config['loss'],
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("=" * 60)
        print("GRU Model Architecture")
        print("=" * 60)
        model.summary()
        print("=" * 60)
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the GRU model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features (samples, timesteps, features)
        y_train : np.ndarray
            Training targets (samples,)
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation targets
            
        Returns:
        --------
        keras.callbacks.History
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.training_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        print("\nTraining GRU model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✓ GRU training complete!")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (samples, timesteps, features)
            
        Returns:
        --------
        np.ndarray
            Predicted volatility values
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        print("\nGRU Model Evaluation:")
        print(f"  MSE: {mse:.8f}")
        print(f"  MAE: {mae:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        print(f"  Directional Accuracy: {directional_accuracy:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"✓ Saved GRU model to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Loaded GRU model from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("GRU Volatility Prediction Model")
    print("This module should be imported and used in the training pipeline.")
