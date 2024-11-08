from flask import Flask, request, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
from models.isolation_forest import IsolationForestModel
from models.vae import VAEModel
from models.gan import GANModel
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
scaler = StandardScaler()
models = {"isolation_forest": IsolationForestModel(), "vae": None, "gan": None}

# Cấu hình Swagger UI
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  # Đường dẫn đến file YAML
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Anomaly Detection API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/train', methods=['POST'])
def train():
    model_type = request.form.get('model_type')
    file = request.files.get('file')
    data = pd.read_csv(file)
    features = data.drop(columns=['target'])
    labels = data['target']

    scaled_features = scaler.fit_transform(features)
    if model_type == "vae":
        models["vae"] = VAEModel(input_dim=scaled_features.shape[1])
        models["vae"].train(scaled_features)
    elif model_type == "gan":
        models["gan"] = GANModel(input_dim=scaled_features.shape[1])
        models["gan"].train(scaled_features)
    else:
        models["isolation_forest"].train(scaled_features)
    return jsonify({"message": f"{model_type} model trained successfully"})

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    file = request.files.get('file')
    data = pd.read_csv(file)
    features = scaler.transform(data.drop(columns=['target']))
    labels = data['target'].values

    model = models.get(model_type)
    if model is None:
        return jsonify({"error": f"Model '{model_type}' has not been trained yet."}), 400

    # Dự đoán bằng mô hình được chọn
    result = model.predict(features, labels)

    # Trả về kết quả dự đoán
    return jsonify({
        "predictions": result["predictions"].tolist(),
        "anomaly_scores": result["anomaly_scores"].tolist(),
        "threshold": result["threshold"],
        "confusion_matrix": result["confusion_matrix"].tolist(),
        "classification_report": result["classification_report"],
        "roc_auc_score": result["roc_auc_score"]
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
