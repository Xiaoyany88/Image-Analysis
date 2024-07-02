# SSY098 Project 1: Image Classification Using Convolutional Neural Networks

## Author
Zhang Xiaoyang

## Abstract
This project aims to develop a Convolutional Neural Network (CNN) model to classify tree bark images from the TRUNK12 dataset, which consists of images from 12 different tree species. The project involves designing a CNN, training it, and implementing improvements such as data normalization, batch normalization, ReLU activation, dropout layers, and introducing more convolutional filters to enhance the model's performance.

## Method

### 1. Network Architecture
The basic CNN architecture used in this project includes:
- Input layer for 40x40x3 color images
- Two convolutional layers with 8 and 16 filters respectively, each followed by max pooling layers
- A fully connected layer with 64 neurons
- An output layer with 12 neurons for 12 classes, followed by a softmax activation layer

### 2. Improvements
Improvements to the model were introduced progressively, including:
- **Data Normalization**: Normalizing images to a mean of 0 and variance of 1 to ensure consistent processing and faster convergence.
- **Increased Convolutional Layers**: Expanding to 3 convolutional layers with filter sizes of 32, 64, and 128.
- **Batch Normalization**: Stabilizing and speeding up the training process.
- **ReLU Activation**: Introducing non-linearity to the model and addressing the vanishing gradient problem.
- **Dropout Layer**: Adding a dropout layer with a dropout rate of 0.5 to prevent overfitting.

## Experimental Evaluation

### Training Parameters
- Optimizer: SGDM
- Initial Learning Rate: 0.01
- Max Epochs: 20
- Shuffle: Every epoch
- Test Frequency: 15 iterations

### Results
#### Basic CNN Model
- Achieved training accuracy close to 80% at epoch 30.
- Test accuracy: 54.88%, indicating overfitting.

#### Basic CNN with Normalized Data
- Improved test accuracy to 57.2%.
- Test loss stabilized at a higher value, indicating overfitting due to increased sensitivity to noise.

#### Enhanced CNN Model with Normalized Data
- Training accuracy: ~90%
- Test accuracy: 81.63%
- More stable training and test loss curves.
- Better performance in the confusion matrix.

## Ablation Study
To demonstrate the importance of each enhancement, several experiments were conducted by removing one improvement at a time:

- **Without Data Normalization**: Test accuracy dropped to 78.21%.
- **Without ReLU Activation**: Test accuracy significantly dropped to 74.68%.
- **Without Dropout Layer**: Test accuracy was 79.2%, with greater overfitting.
- **Without Increased Convolutional Filters**: Test accuracy dropped to 71.5%.

## Conclusion
The enhancements, including data normalization, ReLU activation, dropout layers, and increased convolutional filters, collectively contributed to improved model stability and accuracy. Future work could focus on advanced feature engineering, data augmentation, and experimenting with more complex model architectures to further enhance classification accuracy and generalization.
