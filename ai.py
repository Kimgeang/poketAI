import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import TensorBoard
import datetime







# 1️⃣ ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.4,
    brightness_range=(0.6,1.4),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # 검증용은 증강 없이

# 2️⃣ 학습용/검증용 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Administrator\Desktop\test\dataset\train',  # 이미 학습용으로 분리된 폴더
    target_size=(224,224),
    batch_size=4,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    r'C:\Users\Administrator\Desktop\test\dataset\val',    # 이미 검증용으로 분리된 폴더
    target_size=(224,224),
    batch_size=4,
    class_mode='categorical'
)

# 3️⃣ 사전학습 ResNet50 모델 불러오기
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Base 모델 동결

# 4️⃣ 새 FC층 추가
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # 포켓몬 6종
])

# 5️⃣ 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 샘플 수 확인
print("train samples:", train_generator.samples)
print("val samples:", val_generator.samples)
print("class distribution (val):")
for k,v in val_generator.class_indices.items():
    print(k, "->", sum(1 for fn in os.listdir(os.path.join(val_generator.directory, k))))

# 배치 단위 val loss 분산 확인 (Keras evaluate로 배치 계산)
import numpy as np
val_steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
losses = []
for i in range(val_steps):
    x_batch, y_batch = next(val_generator)
    l = model.test_on_batch(x_batch, y_batch)   # [loss, acc] 형태
    losses.append(l[0])
print("val_batches:", len(losses), " mean:", np.mean(losses), " std:", np.std(losses))



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# 6️⃣ 모델 학습
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    callbacks=[tensorboard_callback]   
)

# 7️⃣ 모델 저장
model.save('pokemon_classifier.h5')
