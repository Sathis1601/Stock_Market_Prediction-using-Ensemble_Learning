---
description: Deploy and run the Streamlit application
---

# Streamlit App Deployment Workflow

## Quick Start (Local Development)

// turbo
1. **Run the Streamlit app**:
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## Access the App

- **Local URL**: http://localhost:8501
- **Network URL**: Use this to access from other devices on the same network

## Stop the App

Press `Ctrl+C` in the terminal where Streamlit is running

## Deploy to Streamlit Cloud (Free Hosting)

### Prerequisites
- GitHub account
- Git repository with your code

### Steps

2. **Initialize Git repository** (if not already done):
```bash
git init
git add .
git commit -m "Add Streamlit app"
```

3. **Create GitHub repository and push**:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

4. **Deploy on Streamlit Cloud**:
   - Visit https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub account
   - Select repository: `YOUR_USERNAME/YOUR_REPO`
   - Set main file path: `app.py`
   - Click "Deploy"

5. **Wait for deployment**:
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

## Deploy Options

### Option 1: Heroku
```bash
# Install Heroku CLI first
heroku login
heroku create your-app-name
git push heroku main
```

### Option 2: Docker
```bash
docker build -t stock-risk-predictor .
docker run -p 8501:8501 stock-risk-predictor
```

## Advanced Configuration

### Custom Domain
- Configure in Streamlit Cloud settings
- Add CNAME record to your DNS

### Environment Variables
- Set in Streamlit Cloud: Settings → Secrets
- Format: TOML (same as `.streamlit/secrets.toml`)

### Resource Limits (Streamlit Cloud Free Tier)
- 1 GB RAM
- 1 CPU core
- Sleeps after 7 days of inactivity

## Troubleshooting

### App won't start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify Streamlit installation
streamlit --version
```

### Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Models not loading
```bash
# Train models first
python src/train.py
```

## Features of the Deployed App

✅ Real-time stock data fetching
✅ Ensemble AI predictions (LSTM, GRU, CNN)
✅ Interactive visualizations
✅ Risk classification
✅ Uncertainty quantification
✅ Technical analysis indicators

## Next Steps

1. Test the app locally
2. Share on Streamlit Cloud for public access
3. Customize theme in `.streamlit/config.toml`
4. Add more features or stock tickers

## Support

For issues:
- Check Streamlit logs in terminal
- Review app/README.md for detailed troubleshooting
- Verify all models are trained
