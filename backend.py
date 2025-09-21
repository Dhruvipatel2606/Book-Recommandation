
import numpy as np
import pandas as pd
import pickle
import os
import subprocess
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flasgger import Swagger

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'apispec_1',
            'route': '/apispec_1.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'uiversion': 3,
    'specs_route': '/apidocs/'
}
Swagger(app, config=swagger_config)

# JSON helpers (must be defined before routes)
import json as _json_mod

def _json_default(obj):
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return None

def json_response(payload, status=200):
    return app.response_class(
        response=_json_mod.dumps(payload, default=_json_default),
        status=status,
        mimetype='application/json'
    )

@app.route('/test')
def test():
    """Test endpoint to verify API is working"""
    return jsonify({
        'status': 'success',
        'message': 'Backend API is working!',
        'endpoints': {
            'test': '/test',
            'search': '/search?title=<book_title>',
            'lowest_price': '/lowest-price',
            'run_analysis': '/run-analysis (POST)'
        }
    })

# Helper function to get book info and recommendations
def get_book_info(title):
    # Load datasets
    df = pd.read_csv('shopping_dataset_clean.csv')
    # Strip spaces for robust matching
    df['title'] = df['title'].str.strip()
    df['author'] = df['author'].str.strip()
    try:
        pivot_df = pd.read_csv('shopping_price_comparison.csv')
    except Exception:
        pivot_df = None
    # Find book row by ISBN10 or normalized title
    isbn_query = title.strip()
    norm_title = title.lower().replace(' ', '').strip()
    book_row = df[(df['isbn10'].astype(str).str.strip() == isbn_query) |
                 (df['title'].str.lower().str.replace(' ', '').str.strip() == norm_title)]
    if book_row.empty:
        return {
            'amazon_title': 'N/A',
            'amazon_author': 'N/A',
            'amazon_isbn': 'N/A',
            'amazon_price': 'N/A',
            'amazon_rating': 'N/A',
            'flipkart_title': 'N/A',
            'flipkart_author': 'N/A',
            'flipkart_isbn': 'N/A',
            'flipkart_price': 'N/A',
            'flipkart_rating': 'N/A',
            'cheapest_platform': 'N/A',
            'predicted_rating': 'N/A',
            'recommendations': []
        }
    # Prices
    amazon_price = None
    flipkart_price = None
    amazon_rating = None
    flipkart_rating = None
    cheapest_platform = None
    predicted_rating = None
    recommendations = []
    if pivot_df is not None:
        pivot_df['title'] = pivot_df['title'].str.strip()
        pivot_df['author'] = pivot_df['author'].str.strip()
        isbn_query = title.strip()
        # Try to match by ISBN10 for both Amazon and Flipkart
        amazon_row = pivot_df[(pivot_df['isbn10'].astype(str).str.strip() == isbn_query) & (pivot_df['Amazon_price'].notna())]
        flipkart_row = pivot_df[(pivot_df['isbn10'].astype(str).str.strip() == isbn_query) & (pivot_df['Flipkart_price'].notna())]
        # If not found by ISBN, try by normalized title (case/space-insensitive)
        if amazon_row.empty:
            amazon_row = pivot_df[pivot_df['title'].str.lower().str.replace(' ', '').str.strip() == title.lower().replace(' ', '').strip()]
            amazon_row = amazon_row[amazon_row['Amazon_price'].notna()]
        if flipkart_row.empty:
            flipkart_row = pivot_df[pivot_df['title'].str.lower().str.replace(' ', '').str.strip() == title.lower().replace(' ', '').strip()]
            flipkart_row = flipkart_row[flipkart_row['Flipkart_price'].notna()]
        # Extract info
        if not amazon_row.empty:
            amazon_price = amazon_row.iloc[0].get('Amazon_price', None)
            amazon_rating = amazon_row.iloc[0].get('Amazon_rating', None)
        if not flipkart_row.empty:
            flipkart_price = flipkart_row.iloc[0].get('Flipkart_price', None)
            flipkart_rating = flipkart_row.iloc[0].get('Flipkart_rating', None)
        # Cheapest platform logic
        if amazon_price is not None and flipkart_price is not None:
            cheapest_platform = 'Amazon' if float(amazon_price) < float(flipkart_price) else 'Flipkart'
        elif amazon_price is not None:
            cheapest_platform = 'Amazon'
        elif flipkart_price is not None:
            cheapest_platform = 'Flipkart'
        # Predicted rating
        try:
            model = pickle.load(open('book_rating_model.pkl', 'rb'))
            features = ['Amazon_price', 'Flipkart_price', 'price_difference', 'company_encoded']
            pred_row = amazon_row if not amazon_row.empty else flipkart_row
            # Only predict if all required columns exist and pred_row is not empty
            if not pred_row.empty and all(f in pred_row.columns for f in features):
                X = pred_row[features].fillna(0)
                pred = model.predict(X)[0]
                predicted_rating = pred
            else:
                predicted_rating = None
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_rating = None
    # Recommendations
    try:
        author = book_row.iloc[0]['author']
        author_books = df[(df['author'] == author) & (df['title'].str.lower() != title.lower())]
        recs = author_books[['title', 'author']].drop_duplicates('title').head(3)
        for _, r in recs.iterrows():
            recommendations.append({'title': r['title'], 'author': r['author']})
        if len(recommendations) < 3:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['title'].fillna(''))
            # Use the index of the matched row instead of exact title equality
            base_index = book_row.index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[base_index], tfidf_matrix).flatten()
            sim_indices = cosine_sim.argsort()[::-1]
            for i in sim_indices:
                if i == base_index:
                    continue
                sim_title = df.iloc[i]['title']
                if isinstance(sim_title, str) and all(sim_title != rec['title'] for rec in recommendations):
                    recommendations.append({'title': df.iloc[i]['title'], 'author': df.iloc[i]['author']})
                if len(recommendations) >= 3:
                    break
    except Exception as e:
        print(f"Recommendations error: {e}")
        # keep whatever recommendations we have so far
    def safe_json(val):
        if val is None:
            return None
        try:
            # Convert numpy scalars to native Python types
            if isinstance(val, np.generic):
                val = val.item()
        except Exception:
            pass
        # Handle pandas/numpy NaN or inf
        try:
            if isinstance(val, float):
                if np.isnan(val) or np.isinf(val):
                    return None
        except Exception:
            pass
        return val
    # Find Amazon and Flipkart book info from df
    isbn_query = title.strip()
    amazon_info = df[(df['isbn10'].astype(str).str.strip() == isbn_query) & (df['company'].str.lower().str.strip() == 'amazon')]
    flipkart_info = df[(df['isbn10'].astype(str).str.strip() == isbn_query) & (df['company'].str.lower().str.strip() == 'flipkart')]
    if amazon_info.empty:
        amazon_info = df[(df['title'].str.lower().str.strip() == title.lower().strip()) & (df['company'].str.lower().str.strip() == 'amazon')]
    if flipkart_info.empty:
        flipkart_info = df[(df['title'].str.lower().str.strip() == title.lower().strip()) & (df['company'].str.lower().str.strip() == 'flipkart')]
    return {
        'amazon_title': amazon_info.iloc[0]['title'] if not amazon_info.empty else 'N/A',
        'amazon_author': amazon_info.iloc[0]['author'] if not amazon_info.empty else 'N/A',
        'amazon_isbn': amazon_info.iloc[0]['isbn10'] if not amazon_info.empty else 'N/A',
        'amazon_price': safe_json(amazon_price) if amazon_price is not None else 'N/A',
        'amazon_rating': safe_json(amazon_rating) if amazon_rating is not None else 'N/A',
        'flipkart_title': flipkart_info.iloc[0]['title'] if not flipkart_info.empty else 'N/A',
        'flipkart_author': flipkart_info.iloc[0]['author'] if not flipkart_info.empty else 'N/A',
        'flipkart_isbn': flipkart_info.iloc[0]['isbn10'] if not flipkart_info.empty else 'N/A',
        'flipkart_price': safe_json(flipkart_price) if flipkart_price is not None else 'N/A',
        'flipkart_rating': safe_json(flipkart_rating) if flipkart_rating is not None else 'N/A',
        'cheapest_platform': safe_json(cheapest_platform) if cheapest_platform is not None else 'N/A',
        'predicted_rating': safe_json(predicted_rating) if predicted_rating is not None else 'N/A',
        'recommendations': recommendations
    }

def lowest_price():
    try:
        # Try to use cleaned dataset if available
        csv_path = 'shopping_dataset_clean.csv' if os.path.exists('shopping_dataset_clean.csv') else 'shopping_dataset.csv'
        df = pd.read_csv(csv_path)
        if 'price' not in df.columns:
            return jsonify({'result': 'No price column found in dataset.'}), 400
        min_row = df.loc[df['price'].idxmin()]
        book_info = f"<b>Title:</b> {min_row.get('title', 'N/A')}<br>"
        book_info += f"<b>Author:</b> {min_row.get('author', 'N/A')}<br>"
        book_info += f"<b>Price:</b> ₹{min_row.get('price', 'N/A')}<br>"
        book_info += f"<b>Company:</b> {min_row.get('company', 'N/A')}<br>"
        return jsonify({'result': book_info})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'}), 500

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    try:
        result = subprocess.run(
            ['python', 'shopping_analysis.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )
        output = result.stdout + '\n' + result.stderr
        return jsonify({'result': output})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'}), 500

@app.route('/lowest-price', methods=['GET'])
def lowest_price_route():
    return lowest_price()

@app.route('/search')
def search():
    """
    Search for a book by title or ISBN10 and return merged platform details
    ---
    parameters:
      - name: title
        in: query
        type: string
        required: true
        description: Book title or ISBN10
    responses:
      200:
        description: Book details and price comparison
      404:
        description: Book not found
      500:
        description: Internal server error
    """
    try:
        title = request.args.get('title', '').strip()
        if not title:
            return json_response({'error': 'No title provided'}, status=400)
        result = get_book_info(title)
        if not result:
            return json_response({'error': 'Book not found'}, status=404)
        return json_response(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json_response({'error': f'Internal server error: {str(e)}'}, status=500)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5026, use_reloader=False)
