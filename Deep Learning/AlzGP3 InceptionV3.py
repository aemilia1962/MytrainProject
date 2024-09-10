# การนำเข้าไลบรารีที่จำเป็นสำหรับการประมวลผลข้อมูล, การวิเคราะห์, และการสร้างโมเดล
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

# ไลบรารีสำหรับการจัดการไฟล์และไดเรกทอรี
import os
from distutils.dir_util import copy_tree, remove_tree

# การจัดการภาพ
from PIL import Image
from random import randint

# ไลบรารีสำหรับการแบ่งข้อมูลและการประเมินผล
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D

# ไลบรารีเสริมสำหรับ TensorFlow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

# กำหนดไดเรกทอรีพื้นฐานสำหรับชุดข้อมูล
base_dir = "V:/Work/RS/Alzheimer Dataset 2/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

# ลบและสร้างใหม่ไดเรกทอรีการทำงานเพื่อเตรียมข้อมูล
if os.path.exists(work_dir):
    remove_tree(work_dir)
os.mkdir(work_dir)
copy_tree(train_dir, work_dir) # คัดลอกข้อมูลฝึกฝน
copy_tree(test_dir, work_dir) # คัดลอกข้อมูลทดสอบ
print("Working Directory Contents:", os.listdir(work_dir))

# กำหนดไดเรกทอรีการทำงานและคลาสของข้อมูล
WORK_DIR = './dataset/'
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 150 # ขนาดภาพ
DIM = (IMG_SIZE, IMG_SIZE)

# กำหนดการเพิ่มข้อมูลเพื่อหลีกเลี่ยงการเรียนรู้ที่เกินจริงและเพิ่มความหลากหลายของชุดข้อมูล
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"
work_dr = IDG(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

# โหลดข้อมูลฝึกฝนจากไดเรกทอรี
train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)

# ดึงข้อมูลจาก ImageDataGenerator
train_data, train_labels = train_data_gen.next()

# แสดงขนาดของชุดข้อมูล
print(train_data.shape, train_labels.shape)

# Reshape the training data for SMOTE
train_data_reshaped = train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3)

# ใช้ SMOTE เพื่อจัดการกับความไม่สมดุลของข้อมูล
sm = SMOTE(random_state=42)
train_data_resampled, train_labels_resampled = sm.fit_resample(train_data_reshaped, train_labels)

# Reshape the data back to the original shape after SMOTE
train_data_resampled = train_data_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# แบ่งข้อมูลเป็นชุดฝึกฝน, ทดสอบ, และตรวจสอบ
train_data_resampled, test_data, train_labels_resampled, test_labels = train_test_split(train_data_resampled, train_labels_resampled, test_size=0.2, random_state=42)
train_data_resampled, val_data, train_labels_resampled, val_labels = train_test_split(train_data_resampled, train_labels_resampled, test_size=0.2, random_state=42)

# โหลดโมเดล InceptionV3 โดยไม่รวมชั้นบนสุด และล็อคการเรียนรู้ของเลเยอร์
inception_base = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
for layer in inception_base.layers:
    layer.trainable = False

# สร้างโมเดลที่รวม InceptionV3
custom_inception_model = Sequential([
    inception_base,
    Dropout(0.5),
    GlobalAveragePooling2D(),
    Flatten(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(4, activation='softmax')        
], name = "inception_cnn_model")


# กำหนดเมตริกสำหรับการประเมินผลโมเดล
METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),  
  tf.keras.metrics.AUC(name='auc')
]

# คอมไพล์โมเดล
custom_inception_model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(), metrics=METRICS)

# แสดงสรุปโมเดล
custom_inception_model.summary()

# ฝึกโมเดลและตรวจสอบโดยใช้ข้อมูลตรวจสอบ
EPOCHS = 20
history = custom_inception_model.fit(
    train_data_resampled, 
    train_labels_resampled, 
    validation_data=(val_data, val_labels), 
    epochs=EPOCHS
)

# ประเมินผลโมเดลบนชุดข้อมูลทดสอบ
test_scores = custom_inception_model.evaluate(test_data, test_labels)
print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))

# ทำนายข้อมูลทดสอบ
pred_labels = custom_inception_model.predict(test_data)

# พล็อตเมทริกซ์ความสับสนเพื่อทำความเข้าใจการจำแนกประเภทอย่างละเอียด
pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)
conf_arr = confusion_matrix(test_ls, pred_ls)
print(classification_report(test_ls, pred_ls))

# แสดงเมทริกซ์ความสับสน
from sklearn.metrics import ConfusionMatrixDisplay
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
disp = ConfusionMatrixDisplay(confusion_matrix=conf_arr, display_labels=labels)
fig, ax = plt.subplots(figsize=(18, 11))
disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='winter')
plt.show()
