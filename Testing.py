import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# Set the path to your dataset and the saved model
MODEL_PATH = 'C:/Users/ADMIN/Documents/ALZHIMERDETECTION/Alzhiemer.h5'
TEST_DATA_PATH = 'C:/Users/ADMIN/Documents/ALZHIMERDETECTION'  # Update this path to your test dataset

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (update according to your dataset's class names)
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Function to preprocess and predict a single image
def preprocess_and_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize image

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = tf.nn.softmax(prediction[0])[pred_class]

    return pred_class, confidence.numpy()

# Function to display the image with prediction results
def display_prediction(img_path, pred_class, confidence):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred_class]} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Function to evaluate the model on the test dataset
def evaluate_model(test_data_path, model):
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_path,
        batch_size=32,
        image_size=(128, 128),
        shuffle=False
    )

    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    predictions = []
    labels = []

    for X, y in test_data:
        y_pred = model.predict(X, verbose=0)
        y_prediction = np.argmax(y_pred, axis=1)
        predictions.extend(y_prediction)
        labels.extend(y)

    print(classification_report(labels, predictions, target_names=class_names))

    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10,6), dpi=300)
    sns.heatmap(cm_df, annot=True, cmap="Greys", fmt=".1f")
    plt.title("Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted", fontweight="bold")
    plt.ylabel("True", fontweight="bold")
    plt.show()

# Set the path to the image you want to test
IMG_PATH = 'C:/Users/ADMIN/Documents/ALZHIMERDETECTION/Non_Demented/non.jpg'

# Perform prediction for a single image
pred_class, confidence = preprocess_and_predict(IMG_PATH, model)

# Display the image and prediction results
display_prediction(IMG_PATH, pred_class, confidence)

# Evaluate the model on the test dataset
evaluate_model(TEST_DATA_PATH, model)
