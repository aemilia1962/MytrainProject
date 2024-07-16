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

# ไลบรารีสำหรับการจัดการกับความไม่สมดุลของข้อมูล
from imblearn.over_sampling import SMOTE
# ไลบรารีสำหรับการแบ่งข้อมูลและการประเมินผล
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix

# ไลบรารีเสริมสำหรับ TensorFlow
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau

# from tensorflow.keras.applications.Densenet import densenet121
from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

# กำหนดไดเรกทอรีพื้นฐานสำหรับชุดข้อมูล
base_dir = "V:/Work/RS/Alzheimer Dataset 2/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

# ลบและสร้างใหม่ไดเรกทอรีการทำงานเพื่อเตรียมข้อมูล ส่วนของโค้ดนี้ทำหน้าที่เตรียมพื้นที่การทำงานสำหรับโมเดลและข้อมูลที่จะถูกใช้งานในกระบวนการเรียนรู้
if os.path.exists(work_dir):
    remove_tree(work_dir)
os.mkdir(work_dir)
copy_tree(train_dir, work_dir) # คัดลอกข้อมูลฝึกฝน
copy_tree(test_dir, work_dir) # คัดลอกข้อมูลทดสอบ
print("Working Directory Contents:", os.listdir(work_dir))

# กำหนดไดเรกทอรีการทำงานและคลาสของข้อมูล
WORK_DIR = './dataset/'
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 176 # ขนาดภาพ
IMAGE_SIZE = [176, 176]
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

# ใช้ SMOTE เพื่อจัดการกับความไม่สมดุลของข้อมูล
sm = SMOTE(random_state=42)
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)
train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(train_data.shape, train_labels.shape)

# แบ่งข้อมูลเป็นชุดฝึกฝน, ทดสอบ, และตรวจสอบ
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

# โหลดโมเดล Densenet121 และกำหนดการฝึกฝน
densenet_model = DenseNet121(input_shape=(176, 176, 3), include_top=False, weights="imagenet")
for layer in densenet_model.layers:
    layer.trainable=False

# สร้างโมเดลปรับแต่งจาก Densenet121
custom_densenet_model = Sequential([
    densenet_model,
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
], name="resnet_cnn_model")

# กำหนดเมตริกสำหรับการประเมินผลโมเดล
METRICS = [
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),  
  tf.keras.metrics.AUC(name='auc')
]

# คอมไพล์โมเดล
custom_densenet_model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(), metrics=METRICS)

# แสดงสรุปโมเดล
custom_densenet_model.summary()

# ฝึกโมเดลและตรวจสอบโดยใช้ข้อมูลตรวจสอบ
EPOCHS = 20
history = custom_densenet_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=EPOCHS)

# ประเมินผลโมเดลบนชุดข้อมูลทดสอบ
test_scores = custom_densenet_model.evaluate(test_data, test_labels)
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))

# ทำนายข้อมูลทดสอบ
pred_labels = custom_densenet_model.predict(test_data)

# พล็อตเมทริกซ์ความสับสนเพื่อทำความเข้าใจการจำแนกประเภทอย่างละเอียด
pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)
conf_arr = confusion_matrix(test_ls, pred_ls)
print(classification_report(test_ls, pred_ls))
conf_arr

# แสดงเมทริกซ์ความสับสน
from sklearn.metrics import ConfusionMatrixDisplay
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
disp = ConfusionMatrixDisplay(confusion_matrix=conf_arr, display_labels=labels)
fig, ax = plt.subplots(figsize=(18, 11))
disp = disp.plot(xticks_rotation='vertical', ax=ax,cmap='winter')
plt.show()
