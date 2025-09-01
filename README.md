Solanum Tuberosum Leaf Disease Classification using Custom convolutional neural network
**Overview:**
This project uses a Custom Convolutional Neural Network (CNN) to classify potato (Solanum tuberosum) leaf conditions into three categories:
Healthy
Early Blight
Late Blight
out of these categories late and early blight are the diseases that can result into complete destruction of potato crops, the main motivation behind the project is to assist farmers financially. in order to make them predict diseases in potato leaves using this module instead of relying on botanist who charges a heavy amount for a single inspection.
Designed for precision agriculture and crop monitoring, the model enables early disease detection and helps prevent yield losses.

**Dataset**
Source: M. A. Putra, Potato Leaf Disease Dataset, Kaggle, 2024, Available:https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
Classes: Healthy, Early Blight, Late Blight

**Preprocessing:**
Resize: 256×256 px
Normalization: pixel values scaled [0,1]
Augmentation: rotation + flipping (training only)

**Model Architecture:**
Frameworks: TensorFlow & Keras
Layers: 6 × Conv2D + MaxPooling → Flatten → Dense (64) → Output (3 neurons, Softmax)
Activations: ReLU (hidden layers), Softmax (output)

**Results:**
The model achieved following results on basis of four performance metrices
Accuracy: 97.01%
Precision: 96.8%
Recall: 96.9%
F1-Score: 96.85%
Below are some results 
[Confusion Matrix](Result_images/Confusion%20matrix.png)  
[Early Blight](Result_images/Early%20blight.png)  
[Late Blight](Result_images/Late%20blight.png)
[Healthy](Result_images/healthy.png)

**Project Workflow:**
**Training & Evaluation:**
The complete training and evaluation workflow is available in the notebook:  
[finalmodel.ipynb](finalmodel.ipynb)
it contains:
Dataset preprocessing
CNN model construction
Training and validation
Performance evaluation (accuracy, precision, recall, F1-score)
**Backend / API:**
[The complete training and evaluation workflow is available in the notebook:  
[test.api.ipynb](test.api.ipynb)
Demonstrates deployment of the trained CNN model using FastAPI.
Input: Potato leaf image
Output: Predicted class (Healthy / Early Blight / Late Blight) with probability score
Shows how this model can be served as a REST API for real-world use
**Frontend**
A simple web-based interface was developed to interact with the FastAPI backend.
Users can upload leaf images.
The model predicts whether the leaf is Healthy, Early Blight, or Late Blight.
Results are displayed with confidence scores.
Note: The frontend source code is not included in this repository due to:
1.Project scope (focusing mainly on the ML model & backend).
2.Codebase privacy.
3.Avoiding unnecessary complexity for recruiters/reviewers.

**Skills Demonstrated:**
Deep Learning (CNNs, TensorFlow/Keras)
Image Preprocessing & Augmentation
Model Evaluation (Accuracy, Precision, Recall, F1)
End-to-End ML Workflow Documentation


