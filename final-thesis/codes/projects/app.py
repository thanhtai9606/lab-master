from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from models.isolation_forest import IsolationForestModel
from models.vae import VAEModel
from models.gan import GANModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

app = Flask(__name__)
api = Api(app, version='1.0', title='API Phát Hiện Bất Thường',
          description='API phát hiện bất thường sử dụng Isolation Forest, VAE, và GAN')

# Khởi tạo namespace
ns = api.namespace('anomaly_detection', description='Các API cho phát hiện bất thường')

scaler = StandardScaler()
models = {
    "isolation_forest": IsolationForestModel(),
    "vae": None,
    "gan": None
}

# Định nghĩa parser cho upload file và các tham số
upload_parser = api.parser()
upload_parser.add_argument('model_type', type=str, required=True, help="Loại mô hình (isolation_forest, vae, gan)")
upload_parser.add_argument('features', type=str, required=True, help="Danh sách các cột đặc trưng, phân cách bằng dấu phẩy")
upload_parser.add_argument('file', type='file', location='files', required=True, help="Tệp CSV chứa dữ liệu")

# Endpoint cho huấn luyện mô hình
@ns.route('/train')
class TrainModel(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Huấn luyện mô hình phát hiện bất thường"""
        args = request.args
        model_type = args['model_type']
        features = args['features'].split(',')
        file = request.files['file']
        data = pd.read_csv(file)

        # Kiểm tra các cột có trong dữ liệu
        for feature in features:
            if feature not in data.columns:
                return {"error": f"Cột '{feature}' không có trong dữ liệu."}, 400

        # Lọc dữ liệu và chuẩn hóa
        data = data[features]
        scaled_features = scaler.fit_transform(data)
        
        if model_type == "vae":
            models["vae"] = VAEModel(input_dim=scaled_features.shape[1])
            models["vae"].train(scaled_features)
        elif model_type == "gan":
            models["gan"] = GANModel(input_dim=scaled_features.shape[1])
            models["gan"].train(scaled_features)
        else:
            models["isolation_forest"].train(scaled_features)
        
        return {"message": f"{model_type} model trained successfully"}

# Endpoint cho dự đoán
@ns.route('/predict')
class PredictModel(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Dự đoán bất thường bằng mô hình đã được huấn luyện"""
        args = request.args
        model_type = args['model_type']
        features = args['features'].split(',')
        file = request.files['file']
        data = pd.read_csv(file)

        # Kiểm tra các cột có trong dữ liệu
        for feature in features:
            if feature not in data.columns:
                return {"error": f"Cột '{feature}' không có trong dữ liệu."}, 400

        # Lọc dữ liệu và chuẩn hóa
        data = data[features]
        scaled_features = scaler.transform(data)

        model = models.get(model_type)
        if model is None:
            return {"error": f"Model '{model_type}' chưa được huấn luyện."}, 400

        # Dự đoán
        result = model.predict(scaled_features)

        return {
            "predictions": result["predictions"].tolist(),
            "anomaly_scores": result["anomaly_scores"].tolist(),
            "threshold": result["threshold"]
        }

# Endpoint kiểm tra trạng thái mô hình
@ns.route('/status')
class ModelStatus(Resource):
    @ns.doc(params={'model_type': 'Loại mô hình (isolation_forest, vae, gan)'})
    def get(self):
        """Kiểm tra trạng thái mô hình"""
        model_type = request.args.get('model_type')
        model = models.get(model_type)
        
        if model is None:
            return {"status": f"{model_type} model chưa được huấn luyện."}
        return {"status": f"{model_type} model đã được huấn luyện và sẵn sàng sử dụng."}

# Endpoint liệt kê các mô hình và trạng thái huấn luyện
@ns.route('/list_models')
class ListModels(Resource):
    def get(self):
        """Liệt kê trạng thái của các mô hình"""
        model_status = {name: "Trained" if model else "Not trained" for name, model in models.items()}
        return model_status

# Endpoint đánh giá mô hình
@ns.route('/evaluate')
class EvaluateModel(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Đánh giá mô hình trên dữ liệu kiểm tra"""
        args = request.args
        model_type = args['model_type']
        features = args['features'].split(',')
        file = request.files['file']
        data = pd.read_csv(file)

        # Kiểm tra các cột có trong dữ liệu
        for feature in features:
            if feature not in data.columns:
                return {"error": f"Cột '{feature}' không có trong dữ liệu."}, 400

        # Lọc dữ liệu và chuẩn hóa
        data = data[features]
        true_labels = data['target'].values if 'target' in data.columns else None
        if true_labels is None:
            return {"error": "Dữ liệu kiểm tra phải chứa cột 'target'."}, 400

        scaled_features = scaler.transform(data)
        model = models.get(model_type)
        if model is None:
            return {"error": f"{model_type} model chưa được huấn luyện."}, 400

        # Đánh giá
        result = model.predict(scaled_features, true_labels)

        return {
            "confusion_matrix": result["confusion_matrix"].tolist(),
            "classification_report": result["classification_report"],
            "roc_auc_score": result["roc_auc_score"]
        }

if __name__ == '__main__':
    app.run(debug=True)
