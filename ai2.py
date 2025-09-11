# 1️⃣ 라이브러리 불러오기
from tensorflow.keras.applications import ResNet50   # ImageNet 사전학습 모델
from tensorflow.keras import layers, models          # 레이어, 모델 구성
from tensorflow.keras.optimizers import Adam         # 옵티마이저
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 데이터 증강
from tensorflow.keras.callbacks import TensorBoard
import datetime




# =========================================
# 2️⃣ 데이터 준비 & 증강
# =========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 0~255 픽셀 값을 0~1로 정규화
    rotation_range=30,       # 이미지 회전
    width_shift_range=0.4,   # 좌우 이동
    height_shift_range=0.4,  # 상하 이동
    horizontal_flip=True     # 좌우 반전
)

val_datagen = ImageDataGenerator(rescale=1./255)  # 검증 데이터는 정규화만

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Administrator\Desktop\test\dataset\train',
    target_size=(224,224),   # ResNet50 입력 크기
    batch_size=8,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    r'C:\Users\Administrator\Desktop\test\dataset\val',
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical'
)

num_pokemon_classes = train_generator.num_classes  # 포켓몬 클래스 수 자동 설정

# =========================================
# 3️⃣ 사전학습 모델 불러오기
# =========================================
base_model = ResNet50(
    weights='imagenet',       # ImageNet으로 학습된 가중치 사용
    include_top=False,         # 기존 출력층 제거
    input_shape=(224,224,3)
)

# =========================================
# 4️⃣ 출력층 구성
# =========================================
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)       # 특징 맵을 1차원 벡터로 변환
x = layers.Dense(1024, activation='relu')(x) # Dense 층으로 포켓몬 특징 학습
predictions = layers.Dense(num_pokemon_classes, activation='softmax')(x)  # 포켓몬 클래스 수 출력층

model = models.Model(inputs=base_model.input, outputs=predictions)  # base_model + 새 출력층 결합

log_dir = "logs/pokemon_finetune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)



# =========================================
# 5️⃣ 초기 학습: 기본 층 동결
# =========================================
for layer in base_model.layers:
    layer.trainable = False  # ImageNet에서 학습된 특징은 그대로 사용

# 모델 컴파일
model.compile(
    optimizer='adam',                         # Adam 옵티마이저
    loss='categorical_crossentropy',          # 다중 클래스 손실
    metrics=['accuracy']                      # 정확도 평가
)

# 출력층만 학습
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[tensorboard_callback]   # TensorBoard 콜백 추가
)

# =========================================
# 6️⃣ Fine-Tuning: 상위 층 일부 학습
# =========================================
for layer in base_model.layers[-10:]:          # 마지막 10개 층만 학습 가능
    layer.trainable = True

# 낮은 학습률로 재컴파일
model.compile(
    optimizer=Adam(1e-5),                      # 안정적 Fine-Tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tuning 학습
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[tensorboard_callback]   # TensorBoard 콜백 추가

)

# =========================================
# 7️⃣ 모델 저장
# =========================================
model.save('pokemon_finetuned_resnet50.h5')   # 학습 완료 모델 저장
