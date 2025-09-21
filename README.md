📚 Book Price Comparison & Recommendation System
A beautiful, modern web application that compares book prices between Amazon and Flipkart, provides smart recommendations, and includes machine learning-based rating predictions.

Features:
🔍 Smart Book Search - Search by title or ISBN
💰 Price Comparison - Compare prices between Amazon and Flipkart
⭐ Rating Analysis - View ratings from both platforms
🎯 AI Recommendations - Get book recommendations based on similarity
🤖 ML Predictions - Machine learning-based rating predictions
📱 Responsive Design - Works perfectly on all devices
🎨 Modern UI - Beautiful gradient design with smooth animations

🚀 Execution Steps:

Method 1: One-Click Startup
```bash
python start_app.py
```
This script will:
- Check and install missing dependencies
- Verify data files
- Start the Flask server automatically

Method 2: Manual Setup
1. Start Backend Server:
   ```bash
   python backend.py
   ```
2. Open Frontend:
   - Navigate to `frontend` folder
   - Open `index.html` in your browser
   - OR access the API directly at `http://localhost:5000`

🔧 Installation

Prerequisites:
- Python 3.7 or higher
- pip (Python package installer)
- Modern web browser
Install Dependencies:
```bash
pip install -r requirements.txt
```

Project Structure
```
├── backend.py                 # Flask backend server
├── frontend/                  # Frontend files
│   ├── index.html            # Main HTML page
│   ├── app.js                # JavaScript functionality
│   └── style.css             # Modern CSS styling
├── start_app.py              # Startup script
├── requirements.txt           # Python dependencies
├── *.csv                     # Data files
└── README.md                 # This file
```

Happy Book Shopping!📚✨
