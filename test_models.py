"""
Quick test to verify models can be loaded
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.lstm_model import LSTMVolatilityModel
from models.gru_model import GRUVolatilityModel
from models.cnn_model import CNNVolatilityModel
from models.ensemble import EnsembleVolatilityPredictor

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

models_dir = Path("saved_models")

print(f"\nChecking {models_dir}/")
print(f"  Files found: {list(models_dir.glob('*.keras'))}")

# Test loading
print("\n[1/3] Loading LSTM model...")
lstm_model = LSTMVolatilityModel()
lstm_model.load_model(str(models_dir / "lstm_model.keras"))
print("  ✅ LSTM loaded successfully")

print("\n[2/3] Loading GRU model...")
gru_model = GRUVolatilityModel()
gru_model.load_model(str(models_dir / "gru_model.keras"))
print("  ✅ GRU loaded successfully")

print("\n[3/3] Loading CNN model...")
cnn_model = CNNVolatilityModel()
cnn_model.load_model(str(models_dir / "cnn_model.keras"))
print("  ✅ CNN loaded successfully")

print("\n[4/4] Creating ensemble...")
ensemble = EnsembleVolatilityPredictor(lstm_model, gru_model, cnn_model)
print("  ✅ Ensemble created successfully")

print("\n" + "=" * 60)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("=" * 60)
print("\nYour Streamlit app should now work properly!")
print("Refresh your browser at: http://localhost:8501")
