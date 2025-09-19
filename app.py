import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import io

app = Flask(__name__)

# -----------------------------
# device 설정 (GPU 사용)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 모델 정의
# -----------------------------
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3)  # 예시 CNN
        self.fc = nn.Linear(16*62*62, 2)  # 2 클래스 예시

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -----------------------------
# 모델 로드 (GPU로 이동, 일부 checkpoint 적용)
# -----------------------------
checkpoint = torch.load('model.pth', map_location=device)
model = MyCNN().to(device)
model_dict = model.state_dict()

# fc 제외하고 checkpoint 불러오기
pretrained_dict = {k: v.to(device) for k, v in checkpoint.items() if k in model_dict and 'fc' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()

# -----------------------------
# 이미지 전처리 함수
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # conv 입력 크기 맞춤
    transforms.ToTensor()
])

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # 배치 차원 추가
    return img.to(device)

# -----------------------------
# 서버 엔드포인트
# -----------------------------


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img_tensor = preprocess_image(file.read())
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    return jsonify({'prediction': pred})

# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
