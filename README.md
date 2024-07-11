

- This project could not have been done without the [tutorial](https://youtu.be/jztwpsIzEGc?si=EYYvCAhanVw8xWEP) by [Nicknochnack](https://github.com/nicknochnack/ImageClassification).
- I used the skillset learned in this tutorial to expand my project and create a Multi-Class Car Image Classifier to distinguish between car manufacturers. 
- This model will differentiate Ford, Telsa and Toyota based on the images provided.
## Importing Directories

```
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
```
# Load Data
```
data_dir = "/Users/brycekan/Downloads/PersonalPythonProjects/MultiClass/MetaData"
```

# Filtering Images and Miscellaneus Files(only including .jpeg images)
```
#Filter out hidden files like .DS_Store
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
```

# Loading Data from Data Directory
```
data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=32, image_size=(256, 256))
```
# Image Preprocessing
- I accumulated 2259 filtered images off google search, equating to 71 batches total.
- Images are labelled to 0,1,or 2 based on their model.
- 0 = Ford
- 1 = Tesla
- 2 = Toyota
```
data = data.map(lambda x, y: (x / 255.0, y))
total_images = sum(1 for _ in data.unbatch())
print(f'Total number of images: {total_images}')

#Batching Images
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(f'Total number of batches: {len(data)}')

#images are labelled to 0,1,or 2 based on their model.
#0 = Ford
#1 = Tesla
#2 = Toyota
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
```
![Screenshot 2024-07-10 at 3 21 45 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/d5dbcd95-4275-49aa-9b2a-8fb23eb1e31d)

# Splitting Data
I split the batches to 80% for Training, 10% for Validation and remaining for Testing
```
data_size = len(data)
train_size = int(data_size * 0.8) #80% training size
val_size = int(data_size * 0.1) #10% validation size  
test_size = data_size - train_size - val_size #Remaining as test size
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
```
![Screenshot 2024-07-10 at 3 25 10 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/ac606ea8-ce7d-4166-8426-2c95eb1494d7)

# Building Deep Learning Model
```
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(workdir), activation='softmax'))  # Adjusted for multi-class classification

model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
```

# Training Model
- Applied early stopping to prevent overfitting.
- Accuracy and loss plateaus @ ~7-10 epochs
```
logdir = '/Users/brycekan/Downloads/PersonalPythonProjects/MultiClass/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

hist = model.fit(train, epochs=15, validation_data=val, callbacks=[tensorboard_callback, early_stopping])
```
![Screenshot 2024-07-10 at 4 31 48 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/577d0667-2de6-414e-9962-e9d28725be73)

# Plotting Accuracy and Loss Curves
```
#Loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
# Accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
```
![Screenshot 2024-07-10 at 4 29 37 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/c0607ce0-07d8-4260-8079-61cf444720c6)
![Screenshot 2024-07-10 at 4 30 30 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/d4219006-edc6-46ef-ae2e-a7cbb7c5ca50)

# Evaluating Model Performance
- This model is 67% accurate
```
accuracy = SparseCategoricalAccuracy()
for batch in test.as_numpy_iterator(): 
    x, y = batch
    yhat = model.predict(x)
    accuracy.update_state(y, yhat)

print(accuracy.result().numpy())
```
![Screenshot 2024-07-10 at 4 36 24 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/bd7a5ed7-1af6-4f14-9567-f32314990c7e)
# Running Model on TestData
- Test Image:
![TestTesla](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/6d8e4949-f849-422c-872e-a0607344d5d2)

```
img = cv2.imread('/Users/brycekan/Downloads/PersonalPythonProjects/MultiClass/TestData/TestTesla.jpeg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
predicted_class = workdir[np.argmax(yhat)]
print(f'Predicted class is {predicted_class}')
```
![Screenshot 2024-07-10 at 4 33 33 PM](https://github.com/brycekan123/Multi-Class-Car-Image-Classifier/assets/119905092/ca914fe1-3ad5-466b-9364-b462d5251f1d)



  
