 # Fruit-Classification
Fruit Image Classification
Fruit Image Classification with AlexNet
This project demonstrates image classification using the AlexNet architecture on a dataset of fruit images. The dataset consists of images of five different fruit classes: apple, banana, grape, mango, and strawberry. The goal is to train a deep learning model to classify these fruits accurately.

Key Features:
## Data Visualization:
Visualizes sample images from each class and displays the distribution of images across different classes.
## Model Architecture:
Implements the AlexNet architecture using TensorFlow/Keras.
## Data Augmentation:
Applies data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping to enhance the diversity of the training data.
## Training:
Trains the model using a training data generator with real-time data augmentation.
## Model Evaluation:
Evaluates the trained model on a separate test dataset to assess its performance in terms of accuracy and loss.
## Predictions:
Loads the trained model to make predictions on new images, demonstrating its practical usage.
## File Structure:
fruit_classification.ipynb: Jupyter Notebook containing the complete code with detailed explanations and visualizations.
README.md: Markdown file providing an overview of the project, instructions for running the code, and any additional information.
alexnet_final_model/: Directory containing the saved model weights after training.
Usage:
Setup Environment: Make sure to have all dependencies installed, including TensorFlow, Keras, Matplotlib, NumPy, and OpenCV.
## Dataset:
Organize the fruit images into training and testing directories (train/ and test/) with subdirectories for each class.
## Run Code:
Execute the code cells in the Jupyter Notebook sequentially to train the model, visualize results, and make predictions.
## Future Enhancements:
Experiment with different architectures and hyperparameters to improve model performance.
Explore additional data augmentation techniques to further enhance the model's robustness.
