import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import testingModel  


trainSetPath = 'images\\trainSet'
Categorys = ['0', '1']
TARGET_SIZE = (128, 128)

def getData(path):
    images = []
    labels = []

    for category in Categorys:
        category_path = os.path.join(path, category)
        label = int(category)
        for name in os.listdir(category_path):
            greyImage = cv2.cvtColor(cv2.imread(os.path.join(category_path, name)), cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(greyImage, TARGET_SIZE)

            images.append(resized_img)
            labels.append(label)

    return images, labels

def normalizer(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label


images, labels = getData(trainSetPath)

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
trainData = dataset.map(normalizer)
trainData = trainData.shuffle(buffer_size=1000).batch(32)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(trainData, epochs=50, validation_data=trainData, callbacks=[early_stopping])


model.save("modelo.h5")


testSetPath = 'images\\testSet'
imagesTest, labelsTest = getData(testSetPath)
testDataset = tf.data.Dataset.from_tensor_slices((imagesTest, labelsTest))
testData = testDataset.map(normalizer).shuffle(buffer_size=1000).batch(32)


testingModel.evaluate_model(model, testData, imagesTest)
