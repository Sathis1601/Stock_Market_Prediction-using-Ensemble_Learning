"""
Stock Market Volatility & Risk Prediction - Streamlit Application
Uncertainty-Aware Ensemble Deep Learning Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection import StockDataCollector
from src.preprocessing import FeatureEngineer
from models.lstm_model import LSTMVolatilityModel
from models.gru_model import GRUVolatilityModel
from models.cnn_model import CNNVolatilityModel
from models.ensemble import EnsembleVolatilityPredictor

# Page Configuration
st.set_page_config(
    page_title="Stock Risk Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Headers */
    h1 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        padding: 20px 0;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #00d4ff;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0aec0;
        font-weight: 500;
    }
    
    /* Risk Badge Styles */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Select box */
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        border-radius: 10px;
    }
    
    /* Loading animation enhancement */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize components
@st.cache_resource
def initialize_models():
    """Initialize and load trained models"""
    try:
        # Check if models exist
        models_dir = Path(__file__).parent.parent / "saved_models"
        
        if not models_dir.exists() or len(list(models_dir.glob("*"))) == 0:
            return None, "Models not found. Please train the models first."
        
        # Initialize models
        lstm_model = LSTMVolatilityModel()
        gru_model = GRUVolatilityModel()
        cnn_model = CNNVolatilityModel()
        
        # Load weights if they exist
        lstm_path = models_dir / "lstm_model.keras"
        gru_path = models_dir / "gru_model.keras"
        cnn_path = models_dir / "cnn_model.keras"
        
        if lstm_path.exists() and gru_path.exists() and cnn_path.exists():
            lstm_model.load_model(str(lstm_path))
            gru_model.load_model(str(gru_path))
            cnn_model.load_model(str(cnn_path))
            
            # Create ensemble
            ensemble = EnsembleVolatilityPredictor(lstm_model, gru_model, cnn_model)
            return ensemble, "Models loaded successfully"
        else:
            return None, "Model weights not found. Please train the models first."
            
    except Exception as e:
        return None, f"Error loading models: {str(e)}"

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, f"No data found for {ticker}"
        
        # Handle MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Standardize column names to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Reset index to get 'date' column for feature engineer
        data = data.reset_index()
        data.rename(columns={'Date': 'date', 'date': 'date'}, inplace=True) # Rename regardless of case
        data['date'] = pd.to_datetime(data['date'])
        
        # Ensure we have the required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in data.columns:
                return None, f"Required column {col} missing from data"
                
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def create_price_chart(df, ticker):
    """Create interactive price chart with candlesticks"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Action', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#10b981' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef4444' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_volatility_chart(predictions_dict, dates):
    """Create volatility prediction chart with uncertainty bands"""
    fig = go.Figure()
    
    ensemble_pred = predictions_dict['ensemble_prediction']
    uncertainty = predictions_dict['uncertainty']
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=dates,
        y=ensemble_pred * 100,  # Convert to percentage
        mode='lines+markers',
        name='Predicted Volatility',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8)
    ))
    
    # Uncertainty bands
    upper_bound = (ensemble_pred + uncertainty) * 100
    lower_bound = (ensemble_pred - uncertainty) * 100
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(0, 212, 255, 0.2)',
        fill='tonexty',
        showlegend=True
    ))
    
    # Risk thresholds
    fig.add_hline(
        y=config['risk_thresholds']['low'] * 100,
        line_dash="dash",
        line_color="#10b981",
        annotation_text="Low Risk Threshold"
    )
    
    fig.add_hline(
        y=config['risk_thresholds']['medium'] * 100,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Medium Risk Threshold"
    )
    
    fig.update_layout(
        title="Volatility Prediction with Uncertainty",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_model_comparison_chart(predictions_dict):
    """Create comparison chart for individual models"""
    models = ['LSTM', 'GRU', 'CNN', 'Ensemble']
    individual_preds = predictions_dict['individual_predictions']
    
    values = [
        individual_preds['lstm'][0] * 100,
        individual_preds['gru'][0] * 100,
        individual_preds['cnn'][0] * 100,
        predictions_dict['ensemble_prediction'][0] * 100
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            marker=dict(
                color=['#8b5cf6', '#ec4899', '#f59e0b', '#00d4ff'],
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=[f'{v:.2f}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Model Predictions Comparison",
        xaxis_title="Model",
        yaxis_title="Predicted Volatility (%)",
        template='plotly_dark',
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        showlegend=False
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence", 'font': {'size': 20, 'color': 'white'}},
        delta={'reference': 70, 'increasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00d4ff"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"}
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center;'>📈 Stock Market Risk Predictor</h1>
        <p style='text-align: center; color: #a0aec0; font-size: 18px;'>
            Uncertainty-Aware Ensemble Deep Learning for Volatility Prediction
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Stock selection
        ticker = st.selectbox(
            "Select Stock",
            config['app']['default_stocks'],
            index=0
        )
        
        # Date range
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.number_input("Days of History", min_value=100, max_value=3650, value=365)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Prediction settings
        st.markdown("### 🎯 Prediction Settings")
        prediction_days = st.slider("Forecast Horizon (days)", 1, 30, 5)
        
        # Display model status
        st.markdown("### 🤖 Model Status")
        ensemble, status_msg = initialize_models()
        
        if ensemble is not None:
            st.success("✅ Models Ready")
        else:
            st.warning(f"⚠️ {status_msg}")
            st.info("To train models, run: `python src/train.py`")
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze Risk", use_container_width=True)
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
            This tool uses an ensemble of:
            - 🔷 **LSTM** - Long-term patterns
            - 🔶 **GRU** - Short-term movements
            - 🔸 **CNN** - Local volatility patterns
            
            **Uncertainty quantification** provides confidence intervals.
        """)
        
        # Disclaimer
        with st.expander("⚠️ Disclaimer"):
            st.markdown("""
                This is an **educational tool** for risk analysis.
                
                **NOT** for direct trading decisions.
                Markets are unpredictable. Always consult
                financial advisors for investment decisions.
            """)
    
    # Main content
    if analyze_button:
        if ensemble is None:
            st.error("❌ Models not available. Please train models first.")
            st.code("python src/train.py", language="bash")
            return
        
        # Fetch data
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            df, error = fetch_stock_data(ticker, start_date, end_date)
            
            if error:
                st.error(f"❌ {error}")
                return
        
        # Display current price info
        st.markdown("## 📊 Current Market Data")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get latest and previous rows
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Extract values (ensuring they are scalars)
        current_price = float(latest['close'])
        prev_price = float(prev['close'])
        volume = float(latest['volume'])
        high = float(latest['high'])
        low = float(latest['low'])
        
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("Volume", f"{volume:,.0f}")
        
        with col3:
            st.metric("Day High", f"${high:.2f}")
        
        with col4:
            st.metric("Day Low", f"${low:.2f}")
        
        # Price chart
        st.markdown("### 📈 Price Action")
        price_fig = create_price_chart(df.tail(config['app']['chart_days']), ticker)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Prepare features
        with st.spinner("🔧 Engineering features..."):
            try:
                feature_engineer = FeatureEngineer()
                df_features = feature_engineer.calculate_all_features(df.copy())
                df_features = feature_engineer.prepare_target_variable(df_features)
                df_features = df_features.dropna()
                
                feature_columns = feature_engineer.get_feature_columns()
                X, y = feature_engineer.create_sequences(df_features, feature_columns)
                
                if len(X) == 0:
                    st.error("❌ Not enough data to make predictions")
                    return
                
            except Exception as e:
                st.error(f"❌ Error processing features: {str(e)}")
                return
        
        # Make predictions
        with st.spinner("🤖 Running ensemble predictions..."):
            try:
                # Use the last sequence for prediction
                X_last = X[-1:] 
                
                predictions = ensemble.predict(X_last)
                risk_level = ensemble.classify_risk(predictions['ensemble_prediction'][0])
                
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
                st.info("This might happen if models weren't trained on compatible data.")
                return
        
        # Display predictions
        st.markdown("## 🎯 Risk Prediction Results")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            volatility_pct = predictions['ensemble_prediction'][0] * 100
            st.metric(
                "Predicted Volatility",
                f"{volatility_pct:.2f}%",
                help="Expected price volatility for the next period"
            )
        
        with col2:
            uncertainty_pct = predictions['uncertainty'][0] * 100
            st.metric(
                "Uncertainty",
                f"±{uncertainty_pct:.2f}%",
                help="Model disagreement indicating prediction uncertainty"
            )
        
        with col3:
            # Risk badge
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f"""
                <div style='text-align: center; padding-top: 10px;'>
                    <p style='color: #a0aec0; font-size: 14px; margin-bottom: 5px;'>RISK LEVEL</p>
                    <div class='{risk_class}'>{risk_level.upper()}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge and model comparison
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            confidence_fig = create_confidence_gauge(predictions['confidence'][0])
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        with col2:
            comparison_fig = create_model_comparison_chart(predictions)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Volatility forecast
        st.markdown("---")
        st.markdown("### 📉 Volatility Forecast")
        
        # Generate future dates for visualization
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # For visualization, we'll repeat the prediction (in reality, you'd retrain with new data)
        vol_predictions = {
            'ensemble_prediction': np.array([predictions['ensemble_prediction'][0]] * prediction_days),
            'uncertainty': np.array([predictions['uncertainty'][0]] * prediction_days)
        }
        
        vol_fig = create_volatility_chart(vol_predictions, future_dates)
        st.plotly_chart(vol_fig, use_container_width=True)
        
        # Individual model details
        with st.expander("🔍 Individual Model Predictions"):
            col1, col2, col3 = st.columns(3)
            
            lstm_pred = predictions['individual_predictions']['lstm'][0] * 100
            gru_pred = predictions['individual_predictions']['gru'][0] * 100
            cnn_pred = predictions['individual_predictions']['cnn'][0] * 100
            
            with col1:
                st.markdown("""
                    <div class='feature-card'>
                        <h4 style='color: #8b5cf6;'>🔷 LSTM Model</h4>
                        <p style='font-size: 24px; font-weight: 700; color: #8b5cf6;'>{:.2f}%</p>
                        <p style='color: #a0aec0;'>Captures long-term dependencies</p>
                    </div>
                """.format(lstm_pred), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='feature-card'>
                        <h4 style='color: #ec4899;'>🔶 GRU Model</h4>
                        <p style='font-size: 24px; font-weight: 700; color: #ec4899;'>{:.2f}%</p>
                        <p style='color: #a0aec0;'>Tracks short-term movements</p>
                    </div>
                """.format(gru_pred), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class='feature-card'>
                        <h4 style='color: #f59e0b;'>🔸 CNN Model</h4>
                        <p style='font-size: 24px; font-weight: 700; color: #f59e0b;'>{:.2f}%</p>
                        <p style='color: #a0aec0;'>Detects local patterns</p>
                    </div>
                """.format(cnn_pred), unsafe_allow_html=True)
        
        # Technical indicators
        with st.expander("📊 Technical Indicators"):
            latest_features = df_features.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RSI (14)", f"{latest_features.get('rsi_14', 0):.2f}")
                st.metric("SMA (20)", f"${latest_features.get('sma_20', 0):.2f}")
            
            with col2:
                st.metric("EMA (12)", f"${latest_features.get('ema_12', 0):.2f}")
                st.metric("EMA (26)", f"${latest_features.get('ema_26', 0):.2f}")
            
            with col3:
                st.metric("MACD", f"{latest_features.get('macd', 0):.4f}")
                st.metric("ATR (14)", f"{latest_features.get('atr_14', 0):.4f}")
            
            with col4:
                st.metric("BB Upper", f"${latest_features.get('bb_upper', 0):.2f}")
                st.metric("BB Lower", f"${latest_features.get('bb_lower', 0):.2f}")
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 💡 Interpretation")
        
        if risk_level == "Low":
            st.success(f"""
                **Low Risk Detected for {ticker}**
                
                The ensemble model predicts relatively stable price movements with low volatility.
                This suggests a more predictable short-term outlook. However, market conditions
                can change rapidly. The uncertainty measure of ±{uncertainty_pct:.2f}% indicates
                the range of model disagreement.
            """)
        elif risk_level == "Medium":
            st.warning(f"""
                **Medium Risk Detected for {ticker}**
                
                The ensemble model predicts moderate volatility ahead. This suggests caution
                and active monitoring. The uncertainty of ±{uncertainty_pct:.2f}% shows some
                model disagreement, which warrants careful consideration.
            """)
        else:
            st.error(f"""
                **High Risk Detected for {ticker}**
                
                The ensemble model predicts elevated volatility in the near term. This suggests
                increased price swings and potential market turbulence. The uncertainty of
                ±{uncertainty_pct:.2f}% indicates the level of model disagreement. Exercise
                extreme caution.
            """)
    
    else:
        # Welcome screen
        st.markdown("""
            <div class='info-card'>
                <h2 style='color: #00d4ff;'>👋 Welcome to the Stock Market Risk Predictor!</h2>
                <p style='font-size: 16px; line-height: 1.6;'>
                    This advanced tool leverages <strong>ensemble deep learning</strong> to predict
                    stock market volatility and quantify prediction uncertainty.
                </p>
                <br>
                <h3 style='color: #ffffff;'>🎯 How It Works:</h3>
                <ol style='font-size: 16px; line-height: 1.8;'>
                    <li>Select a stock ticker from the sidebar</li>
                    <li>Choose your analysis parameters</li>
                    <li>Click <strong>"Analyze Risk"</strong> to generate predictions</li>
                    <li>Review the comprehensive risk assessment</li>
                </ol>
                <br>
                <h3 style='color: #ffffff;'>✨ Key Features:</h3>
                <ul style='font-size: 16px; line-height: 1.8;'>
                    <li>🤖 <strong>Ensemble AI</strong>: Combines LSTM, GRU, and CNN models</li>
                    <li>📊 <strong>Uncertainty Quantification</strong>: Know the confidence level</li>
                    <li>📈 <strong>Interactive Charts</strong>: Visualize price action and volatility</li>
                    <li>🎯 <strong>Risk Classification</strong>: Clear Low/Medium/High risk levels</li>
                    <li>⚡ <strong>Real-time Data</strong>: Powered by Yahoo Finance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("---")
        st.markdown("## 🚀 Model Architecture")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='feature-card'>
                    <h3 style='color: #8b5cf6;'>🔷 LSTM</h3>
                    <p>Long Short-Term Memory networks excel at capturing long-range
                    temporal dependencies in time-series data.</p>
                    <br>
                    <strong>Best for:</strong> Long-term trends
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='feature-card'>
                    <h3 style='color: #ec4899;'>🔶 GRU</h3>
                    <p>Gated Recurrent Units provide efficient modeling of
                    short-term market movements and recent patterns.</p>
                    <br>
                    <strong>Best for:</strong> Short-term patterns
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='feature-card'>
                    <h3 style='color: #f59e0b;'>🔸 CNN</h3>
                    <p>Convolutional Neural Networks extract local volatility
                    patterns and detect sudden market shifts.</p>
                    <br>
                    <strong>Best for:</strong> Pattern recognition
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👈 **Get started** by selecting a stock and clicking 'Analyze Risk' in the sidebar!")

if __name__ == "__main__":
    main()
