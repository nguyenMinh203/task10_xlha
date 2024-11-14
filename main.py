import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn tới dữ liệu
train_dir = "C:/Users/Admin/PycharmProjects/task10_xlha/input/Train"
validation_dir = "C:/Users/Admin/PycharmProjects/task10_xlha/input/Validation"

# Tải dữ liệu và tiền xử lý
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# Tiền xử lý: Lấy một batch dữ liệu từ train và validation generator
train_images, train_labels = next(train_generator)
validation_images, validation_labels = next(validation_generator)

# Chuẩn bị dữ liệu cho SVM: Flatten dữ liệu hình ảnh
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
validation_images_flattened = validation_images.reshape(validation_images.shape[0], -1)

# Chuẩn bị dữ liệu cho ANN và CNN: Chuyển nhãn thành one-hot encoding
train_labels_cat = to_categorical(train_labels, 2)
validation_labels_cat = to_categorical(validation_labels, 2)

# ------------------ SVM ------------------
# Tạo và huấn luyện mô hình SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(train_images_flattened, train_labels)

# Dự đoán và đánh giá mô hình SVM
svm_predictions = svm_model.predict(validation_images_flattened)
svm_accuracy = accuracy_score(validation_labels, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Ma trận nhầm lẫn và báo cáo cho SVM
def plot_confusion_matrix(true_labels, predicted_labels, model_name):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(validation_labels, svm_predictions, 'SVM')

# ------------------ ANN ------------------
# Xây dựng mô hình ANN
ann_model = Sequential([
    Flatten(input_shape=(150, 150, 3)),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_ann = ann_model.fit(train_images, train_labels_cat, epochs=10, batch_size=32, validation_data=(validation_images, validation_labels_cat))

# Vẽ đồ thị độ chính xác
def plot_history(history, model_name):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history_ann, 'ANN')

# Dự đoán và đánh giá mô hình ANN
ann_predictions = np.argmax(ann_model.predict(validation_images), axis=1)
ann_accuracy = accuracy_score(np.argmax(validation_labels_cat, axis=1), ann_predictions)
print("ANN Accuracy:", ann_accuracy)

plot_confusion_matrix(np.argmax(validation_labels_cat, axis=1), ann_predictions, 'ANN')

# ------------------ CNN ------------------
# Xây dựng mô hình CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn_model.fit(train_images, train_labels_cat, epochs=10, batch_size=32, validation_data=(validation_images, validation_labels_cat))

# Vẽ đồ thị độ chính xác cho CNN
plot_history(history_cnn, 'CNN')

# Dự đoán và đánh giá mô hình CNN
cnn_predictions = np.argmax(cnn_model.predict(validation_images), axis=1)
cnn_accuracy = accuracy_score(np.argmax(validation_labels_cat, axis=1), cnn_predictions)
print("CNN Accuracy:", cnn_accuracy)

plot_confusion_matrix(np.argmax(validation_labels_cat, axis=1), cnn_predictions, 'CNN')

# ------------------ In ra báo cáo phân loại ------------------
print("SVM Classification Report:")
print(classification_report(validation_labels, svm_predictions, target_names=['Dog', 'Cat']))

print("ANN Classification Report:")
print(classification_report(np.argmax(validation_labels_cat, axis=1), ann_predictions, target_names=['Dog', 'Cat']))

print("CNN Classification Report:")
print(classification_report(np.argmax(validation_labels_cat, axis=1), cnn_predictions, target_names=['Dog', 'Cat']))
