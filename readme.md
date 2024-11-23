# Medical Image Classification Project

This project implements a Convolutional Neural Network (CNN) to classify medical images into two categories: **Benign** and **Malignant**. The model is trained using a custom dataset and evaluated using performance metrics such as accuracy, precision, recall, and F1-score. Evaluation includes visualizing the predictions and comparing them with the actual labels.

## Description

This project uses TensorFlow and Keras to build, train, and evaluate the image classification model. The model is based on a Convolutional Neural Network (CNN), which is effective for image processing and classification tasks. The dataset is organized into two categories:

- **Benign**: Images representing benign lesions.
- **Malignant**: Images representing malignant lesions.

The process involves the following steps:

1. Image preprocessing.
2. Training the CNN model.
3. Evaluating the model with visual results.
4. Computing performance metrics like accuracy, precision, recall, and F1-score.

## Project Structure

The project consists of the following files:

- `app.py`: Main file that loads the data, trains the model, and saves it.
- `testingModel.py`: Script for evaluating the trained model on a test set and visualizing the results.
- `model.py`: Defines the structure of the CNN model.
- `images/`: Directory containing the dataset of medical images organized into subfolders (`0` for Benign, `1` for Malignant).

## Requirements

To run the project, you'll need the following Python libraries:

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Scikit-learn

You can install them using `pip`:

```bash
pip install tensorflow opencv-python matplotlib numpy scikit-learn
```

## Dataset
images/
    trainSet/
        0/  # Benign images
        1/  # Malignant images
    testSet/
        0/  # Benign images for testing
        1/  # Malignant images for testing

The images used in this project were taken from the [Breast Histopathology Images dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

## Model Architecture
The model consists of the following layers:

- Conv2D Layer: Convolutional layer with 32 filters and a 3x3 kernel, followed by ReLU activation.
- MaxPooling2D Layer: Pooling layer to downsample the feature maps.
- Conv2D Layer: Second convolutional layer with 64 filters and a 3x3 kernel, followed by ReLU activation.
- MaxPooling2D Layer: Pooling layer to downsample the feature maps.
- Conv2D Layer: Third convolutional layer with 128 filters and a 3x3 kernel, followed by ReLU activation.
- MaxPooling2D Layer: Pooling layer to downsample the feature maps.
- Flatten Layer: Flattens the 2D feature maps into a 1D vector.
- Dense Layer: Fully connected layer with 256 neurons and ReLU activation.
- Dropout Layer: Dropout regularization with a rate of 0.5 to prevent overfitting.
- Dense Layer: Output layer with 2 neurons and softmax activation, corresponding to the two classes (Benign and Malignant).

## Visualizing Results
The evaluation script also provides visualizations of the model's predictions:

1. Image Display: Each image is shown along with the predicted label and confidence score.
2. Prediction Bar Chart: A bar chart shows the model's prediction probabilities for each class (Benign and Malignant).




