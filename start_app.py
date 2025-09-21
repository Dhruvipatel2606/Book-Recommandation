#!/usr/bin/env python3
"""
Startup script for Book Price Comparison Application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'numpy', 
        'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstalling missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All required packages are installed!")
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'shopping_dataset_clean.csv',
        'shopping_price_comparison.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Some data files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("   The application may not work properly without these files.")
    
    return True

def main():
    print("🚀 Starting Book Price Comparison Application...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check data files
    check_data_files()
    
    print("\n🌐 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔍 Or open the frontend/index.html file directly")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from backend import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == '__main__':
    main()
