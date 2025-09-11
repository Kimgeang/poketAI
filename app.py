from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 1. 학습된 모델 로드
# model = tf.keras.models.load_model('model/pokemon_like_model.h5')

# 2. 이미지 전처리 함수
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)/255.0
    return x

# 3. 메인 페이지 (이미지 업로드)
@app.route('/')
def index():
    return render_template('index.html')

# 4. 이미지 업로드 후 예측
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    x = preprocess_img(filepath)
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    
    # 클래스 이름 mapping 필요 (train_generator.class_indices 사용)
    class_names = ['pikachu_like','charizard_like']
    result = class_names[class_idx]

    return jsonify({"result": result, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)