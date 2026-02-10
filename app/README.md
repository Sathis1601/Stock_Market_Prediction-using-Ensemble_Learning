# Stock Market Risk Predictor - Streamlit App

## 🚀 Quick Start

### Running Locally

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**:
   - Open your browser and navigate to `http://localhost:8501`
   - The app should open automatically

## 📋 Prerequisites

Before running the app, you have two options:

### Option 1: Use Pre-trained Models (Recommended for Demo)
If you have already trained models, ensure the following files exist:
- `saved_models/lstm_model.h5`
- `saved_models/gru_model.h5`
- `saved_models/cnn_model.h5`

### Option 2: Train Models First
If you haven't trained the models yet:
```bash
python src/train.py
```

**Note**: The app will work in demo mode even without trained models, but predictions won't be available.

## 🌐 Deployment Options

### 1. Local Deployment (Development)
```bash
streamlit run app.py
```

### 2. Streamlit Cloud (Free Hosting)

#### Steps:
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path: `app.py`
   - Click "Deploy"

3. **Configuration**:
   - Streamlit Cloud will automatically detect `requirements.txt`
   - No additional configuration needed

### 3. Docker Deployment

Create a `Dockerfile` in the root directory:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t stock-risk-predictor .
docker run -p 8501:8501 stock-risk-predictor
```

### 4. Heroku Deployment

1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 🎨 Features

- **📊 Real-time Data**: Fetches live stock data from Yahoo Finance
- **🤖 Ensemble AI**: Combines LSTM, GRU, and CNN predictions
- **📈 Interactive Charts**: Candlestick charts, volume analysis
- **🎯 Risk Classification**: Low/Medium/High risk levels
- **📉 Volatility Forecasting**: Predicts future market volatility
- **🔍 Uncertainty Quantification**: Model confidence intervals
- **💡 Technical Indicators**: RSI, MACD, Bollinger Bands, and more

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Model architectures
- Training parameters
- Risk thresholds
- Default stock tickers
- Technical indicators

## 📱 Usage

1. **Select a Stock**: Choose from popular stocks or indices
2. **Set Parameters**: Adjust date range and forecast horizon
3. **Analyze Risk**: Click the button to generate predictions
4. **Review Results**: Examine volatility forecasts, risk levels, and confidence scores

## 🔒 Disclaimer

This tool is for **educational and analytical purposes only**.

- NOT intended for direct trading or investment advice
- Markets are inherently unpredictable
- Always consult financial professionals for investment decisions
- Past performance does not guarantee future results

## 🐛 Troubleshooting

### App won't start
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.8+): `python --version`

### No predictions available
- Train models first: `python src/train.py`
- Check if model files exist in `saved_models/`

### Data fetch errors
- Check internet connection
- Verify ticker symbol is valid
- Try a different date range

## 📞 Support

For issues or questions:
1. Check the error message in the app
2. Review the console output
3. Verify all dependencies are installed
4. Ensure models are trained

## 📄 License

MIT License - Feel free to use and modify for educational purposes.
