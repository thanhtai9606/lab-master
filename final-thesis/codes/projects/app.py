from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from models.isolation_forest import IsolationForestModel
from models.vae import VAEModel
from models.gan import GANModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
api = Api(app, version='1.0', title='API Phát Hiện Bất Thường',
          description='API phát hiện bất thường sử dụng Isolation Forest, VAE, và GAN')

ns = api.namespace('anomaly_detection', description='Các API cho phát hiện bất thường')

scaler = StandardScaler()
scaler_fitted = False  # Cờ kiểm tra scaler đã được fit
models = {
    "isolation_forest": None,
    "vae": None,
    "gan": None
}

upload_parser = api.parser()
upload_parser.add_argument('model_type', type=str, required=True, help="Loại mô hình (isolation_forest, vae, gan)")
upload_parser.add_argument('features', type=str, required=True, help="Danh sách các cột đặc trưng, phân cách bằng dấu phẩy")
upload_parser.add_argument('file', type='file', location='files', required=True, help="Tệp CSV chứa dữ liệu")

@ns.route('/train')
class TrainModel(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Huấn luyện mô hình phát hiện bất thường"""
        global scaler_fitted
        args = request.args
        model_type = args['model_type']
        features = args['features'].split(',')
        file = request.files['file']
        data = pd.read_csv(file)

        for feature in features:
            if feature not in data.columns:
                return {"error": f"Cột '{feature}' không có trong dữ liệu."}, 400

        data = data[features]
        global scaler
        scaled_features = scaler.fit_transform(data)
        scaler_fitted = True  # Đánh dấu scaler đã được fit
        
        if model_type == "vae":
            models["vae"] = VAEModel(input_dim=scaled_features.shape[1])
            models["vae"].train(scaled_features)
        elif model_type == "gan":
            models["gan"] = GANModel(input_dim=scaled_features.shape[1])
            models["gan"].train(scaled_features)
        elif model_type == "isolation_forest":
            models["isolation_forest"] = IsolationForestModel()
            models["isolation_forest"].train(scaled_features)
        else:
            return {"error": f"Loại mô hình '{model_type}' không hợp lệ"}, 400
        
        return {"message": f"{model_type} model trained successfully"}

@ns.route('/predict')
class PredictModel(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Dự đoán bất thường bằng mô hình đã được huấn luyện"""
        global scaler_fitted
        if not scaler_fitted:
            return {"error": "Scaler chưa được fit. Hãy huấn luyện dữ liệu trước."}, 400

        args = request.args
        model_type = args['model_type']
        features = args['features'].split(',')
        file = request.files['file']
        data = pd.read_csv(file)

        for feature in features:
            if feature not in data.columns:
                return {"error": f"Cột '{feature}' không có trong dữ liệu."}, 400

        data = data[features]
        scaled_features = scaler.transform(data)

        model = models.get(model_type)
        if model is None:
            return {"error": f"Model '{model_type}' chưa được huấn luyện."}, 400

        # Dự đoán không cần nhãn thật
        try:
            result = model.predict(scaled_features)
        except Exception as e:
            return {"error": f"Lỗi khi dự đoán với model '{model_type}': {str(e)}"}, 500

        return {
            "predictions": result["predictions"],
            "anomaly_scores": result["anomaly_scores"],
            "threshold": result["threshold"]
        }

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

@ns.route('/list_models')
class ListModels(Resource):
    def get(self):
        """Liệt kê trạng thái của các mô hình"""
        model_status = {name: "Trained" if model else "Not trained" for name, model in models.items()}
        return model_status

if __name__ == '__main__':
    app.run(debug=True)