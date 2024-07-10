import tensorflow as tf
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

data_dir = "/Users/brycekan/Downloads/MultiClass/MetaData"

# Filter out hidden files like .DS_Store
workdir = [f for f in os.listdir(data_dir) if not f.startswith('.')]

for image_class in workdir: 
    image_class_dir = os.path.join(data_dir, image_class)
    if not os.path.isdir(image_class_dir):
        continue

    for image in os.listdir(image_class_dir):
        image_path = os.path.join(image_class_dir, image)
        try: 
            img = cv2.imread(image_path)
            if not image_path.lower().endswith(('.jpeg')):
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))

# Loading data
data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=32, image_size=(256, 256))

# Image preprocessing
data = data.map(lambda x, y: (x / 255.0, y))
total_images = sum(1 for _ in data.unbatch())
print(f'Total number of images: {total_images}')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(f'Total number of batches: {len(data)}')


# Split data
data_size = len(data)
train_size = int(data_size * 0.8) #80% training
val_size = int(data_size * 0.1)   #10% validation
test_size = data_size - train_size - val_size #10% Test

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Building DeepLearningModel
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(workdir), activation='softmax'))  # Multi-Class Classification
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

# Model training
logdir = '/Users/brycekan/Downloads/MultiClass/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# Applied early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback, early_stopping])

# Plotting loss curves
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Plotting accuracy curves
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluating model
accuracy = SparseCategoricalAccuracy()
for batch in test.as_numpy_iterator(): 
    x, y = batch
    yhat = model.predict(x)
    accuracy.update_state(y, yhat)

print(accuracy.result().numpy())

# Running Model on TestData
img = cv2.imread('/Users/brycekan/Downloads/TeslavsToyotaImageClassifier/TestData/TestTesla.jpeg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
predicted_class = workdir[np.argmax(yhat)]
print(f'Predicted class is {predicted_class}')
