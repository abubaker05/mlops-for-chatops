from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import joblib
from backend.ml.trainer import train_model
from deploy import deploy_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, '..', 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/api/datasets/<filename>/columns', methods=['GET'])
def get_dataset_columns(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Dataset not found'}), 404
    try:
        df = pd.read_csv(path, nrows=1)
        return jsonify({'columns': df.columns.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
    return jsonify({'datasets': files})

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    # ... (this function remains the same)
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    df = pd.read_csv(path)
    stats = {
        'rows': int(df.shape[0]), 'cols': int(df.shape[1]),
        'columns': df.columns.tolist()
    }
    return jsonify({'filename': filename, 'stats': stats})


@app.route('/api/train', methods=['POST'])
def train():
    try:
        payload = request.json
        dataset = payload.get('dataset')
        target = payload.get('target')
        
        # <<< --- MODIFIED DICTIONARY --- >>>
        model_name_map = {
            "Linear Regression": "linear_regression",
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree", # Changed from Random Forest
            "SVM": "svm",
            "LightGBM": "lightgbm"
        }
        model_display_name = payload.get('model_type')
        model_type = model_name_map.get(model_display_name)
        
        features = payload.get('features')

        if not dataset or not target or not model_type:
            return jsonify({'error': f'Dataset, target, or model type is missing or invalid. Received: {model_display_name}'}), 400

        dataset_path = os.path.join(UPLOAD_DIR, dataset)
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'dataset not found'}), 404

        df = pd.read_csv(dataset_path)
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found in dataset'}), 400

        X = df.drop(columns=[target]) if not features else df[features]
        y = df[target]
        
        if y.isnull().any():
            return jsonify({'error': 'Target column contains missing values'}), 400
        
        model, metrics = train_model(X, y, model_type)
        model_name = f"{model_type}_{os.path.splitext(dataset)[0]}.joblib"
        model_path = os.path.join(MODELS_DIR, model_name)
        joblib.dump(model, model_path)

        return jsonify({
            'model_name': model_name, 'metrics': metrics,
            'dataset_size': len(df), 'features_used': list(X.columns)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# ... (other routes remain the same) ...

##if __name__ == '__main__':
    ##app.run(port=5001, debug=True)
