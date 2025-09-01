Solanum Tuberosum Leaf Disease Classification using Custom convolutional neural network
**Overview:**
This project uses a Custom Convolutional Neural Network (CNN) to classify potato (Solanum tuberosum) leaf conditions into three categories:
Healthy
Early Blight
Late Blight
Designed for precision agriculture and crop monitoring, the model enables early disease detection and helps prevent yield losses.

**Dataset**
Source: M. A. Putra, Potato Leaf Disease Dataset, Kaggle, 2024, Available:https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
Classes: Healthy, Early Blight, Late Blight

**Preprocessing**
Resize: 256×256 px
Normalization: pixel values scaled [0,1]
Augmentation: rotation + flipping (training only)

**Model Architecture**
Frameworks: TensorFlow & Keras
Layers: 6 × Conv2D + MaxPooling → Flatten → Dense (64) → Output (3 neurons, Softmax)
Activations: ReLU (hidden layers), Softmax (output)

**Results**
Accuracy: 97.01%
Precision: 96.8%
Recall: 96.9%
F1-Score: 96.85%

**Project Workflow**
**Training & Evaluation**
The complete training and evaluation workflow is available in the notebook:  
[finalmodel.ipynb](finalmodel.ipynb)
it contains:
Dataset preprocessing
CNN model construction
Training and validation
Performance evaluation (accuracy, precision, recall, F1-score)
**Backend / API**
[The complete training and evaluation workflow is available in the notebook:  
[test.api.ipynb](test.api.ipynb)
Demonstrates deployment of the trained CNN model using FastAPI.
Input: Potato leaf image
Output: Predicted class (Healthy / Early Blight / Late Blight) with probability score
Shows how this model can be served as a REST API for real-world use

**Skills Demonstrated**
Deep Learning (CNNs, TensorFlow/Keras)
Image Preprocessing & Augmentation
Model Evaluation (Accuracy, Precision, Recall, F1)
End-to-End ML Workflow Documentation


