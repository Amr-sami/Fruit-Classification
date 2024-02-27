import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau

# Path to train and test directories
train_path = 'F:\\imgclass\\train\\'
test_path = 'F:\\imgclass\\test\\'

# Load and display an image
img = load_img(train_path + "1//8.jpeg", target_size=(227, 227))
plt.imshow(img)
plt.axis("off")
plt.show()

# Convert image to array
x = img_to_array(img)
print(x.shape)

# Visualize more images from each class
images = ['1', '2', '3', '4', '5']
fig = plt.figure(figsize=(10, 5))
for i in range(len(images)):
    ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    plt.title(images[i])
    plt.axis("off")

    # Load and display the first image from each class
    class_folder = images[i]
    image_path = os.path.join(train_path, class_folder, "1.jpeg")
    ax.imshow(load_img(image_path, target_size=(227, 227)))
plt.show()

# Count images in each class
image_count = []
class_names = []
print('{:18s}'.format('class'), end='')
print('Count:')
print('-' * 24)
for folder in os.listdir(train_path):
    folder_num = len(glob(os.path.join(train_path, folder, "*.jpeg"))) + \
        len(glob(os.path.join(train_path, folder, "*.png")))
    image_count.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
print('-' * 24)
print("Number of classes : ", len(class_names))

# Plot the number of images in each class
sns.set(rc={'figure.figsize': (5, 5)})
sns.barplot(x=class_names, y=image_count)
plt.ylabel('Number of images in each class')
plt.show()

# Number of classes
num_classes = len(glob(train_path + '/*'))

# Define the AlexNet architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
        4, 4), activation='relu', input_shape=(227, 227, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(
        5, 5), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)

# Summary of the model
model.summary()

# Image data generator
epochs = 40
batch_size = 16
image_height = 227
image_width = 227

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

# Define callbacks
callbacks_list = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.1, patience=10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='alexnet_final_model',
        monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=test_generator,
    verbose=1,
    validation_steps=test_generator.samples // batch_size
)

# Plot model accuracy
plt.figure(1, figsize=(10, 10))
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')

# Plot model loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Evaluate the model
score = model.evaluate(test_generator)
print('test loss:', score[0])
print('test accuracy:', score[1])

# Save the model
model.save("alexnet_final_model")

# Load the saved model
model = tf.keras.models.load_model("alexnet_final_model")

# Function to load and preprocess an image


def load_image(filename):
    img = load_img(filename, grayscale=False,
                   color_mode="rgb", target_size=(227, 227, 3))
    img = img_to_array(img)
    img = img.reshape(1, 227, 227, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img


# Load and predict a sample image
sample_path = 'F:\\imgclass\\predict\\5\\'
img = load_img(sample_path + "7.jpg", target_size=(227, 227))
plt.imshow(img)
plt.axis("off")
plt.show()

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predict_x = model.predict(img_array)
result = np.argmax(predict_x, axis=1)

class_names = {0: 'Apple', 1: 'Banana',
               2: 'Grape', 3: 'Mango', 4: 'Strawberry'}

predicted_class = class_names.get(result[0], "Not in the list")
print(predicted_class)
