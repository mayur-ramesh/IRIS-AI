from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import traceback
import os

# Import the IRIS_System from the existing MVP script
try:
    from iris_mvp import IRIS_System
    iris_app = IRIS_System()
except ImportError as e:
    print(f"Error importing iris_mvp: {e}")
    iris_app = None

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['GET'])
def analyze_ticker():
    """API endpoint to analyze a specific ticker."""
    if not iris_app:
        return jsonify({"error": "IRIS System failed to initialize on the server."}), 500

    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    ticker = str(ticker).strip().upper()

    try:
        print(f"API Request for Analysis: {ticker}")
        # Run the analysis for the single ticker quietly
        report = iris_app.run_one_ticker(ticker, quiet=True)
        
        if report:
            return jsonify(report)
        else:
             return jsonify({"error": f"Failed to analyze {ticker}. Stock not found or connection error."}), 404
             
    except Exception as e:
        print(f"Error during analysis: {traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500

@app.route('/api/chart')
def get_chart():
    """Serve the generated chart image."""
    path = request.args.get('path')
    if not path:
        return jsonify({"error": "No path provided"}), 400
    
    # Basic security check
    if not str(path).startswith('data'):
        return jsonify({"error": "Invalid path"}), 403
        
    full_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        return jsonify({"error": "Chart not found"}), 404
        
    return send_file(full_path, mimetype='image/png')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
