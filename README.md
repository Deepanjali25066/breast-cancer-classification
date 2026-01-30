🩺 Breast Cancer Classification using Deep Learning

📌 Project Overview
Breast cancer is one of the most common cancers among women worldwide. Early and accurate detection plays a crucial role in improving survival rates.
This project focuses on binary classification of breast cancer images (Benign vs Malignant) using deep learning and transfer learning techniques to assist radiologists with faster and more reliable diagnosis.
Multiple state-of-the-art models were implemented, fine-tuned, and compared to identify the most effective architecture.

🎯 Objectives
Build an automated breast cancer image classification system
Apply transfer learning using pretrained models
Compare CNN, Transformer, and Hybrid architectures
Improve accuracy while handling class imbalance
Visualize and analyze model performance

🧠 Models Implemented
🔹 1. ResNet50
Pretrained on ImageNet (frozen base layers)
Added:
Global Average Pooling
Batch Normalization
Dense (256, ReLU)
Dropout (0.5)
Dense (1, Sigmoid)
Optimizer: Adam (LR = 0.0005)
Loss: Binary Cross-Entropy
Metrics: Accuracy, Precision, Recall

🔹 2. EfficientNetV2B0
Transfer learning with ImageNet weights
Custom classification head with Dropout & BatchNorm
Fine-tuning:
First 140 layers frozen
Low learning rate (5e-5)
Callbacks:
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
Class weights used to handle imbalance
Fallback model: MobileNetV2 (if loading fails)

🔹 3. Vision Transformer (ViT-B16)
Pretrained on ImageNet21k + ImageNet2012
Transformer-based global feature extraction
Custom dense head with BatchNorm & Dropout
Optimizer: Adam (LR = 0.0005)
Loss: Binary Cross-Entropy
Fallback: MobileNetV2

🔹 4. DenseNet201
Deep CNN with dense connectivity
Used for comparative evaluation with other architectures

🔹 5. Hybrid Model (EfficientNetV2 + ViT)
CNN + Transformer fusion model
EfficientNetV2 captures local spatial features
ViT captures global contextual features
Feature concatenation followed by dense layers
Final Sigmoid output for binary classification

📊 Results & Performance
All models evaluated using:
Accuracy
Precision
Recall
Hybrid model showed strong performance by combining CNN and Transformer strengths
Visualizations included:
Training vs Validation curves
Model performance comparison charts
(Exact metrics can be found in the notebook and presentation)

🗂️ Dataset & Preprocessing
Medical image dataset for breast cancer classification
Image resizing and normalization
Train-validation split
Class imbalance handled using class weights
Data augmentation applied for better generalization

🛠️ Technologies Used
Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
Transfer Learning
Vision Transformers

📁 Repository Structure
breast-cancer-classification/
│
├── notebook/
│   └── breat-cancer-classification-4-3.ipynb
│
├── README.md

🚀 Future Improvements
Use larger and more diverse datasets
Apply explainable AI techniques (Grad-CAM)
Optimize inference speed for real-time use
Deploy model using Flask / Streamlit

👩‍💻 Author
Deepanjali Thakur
M.Sc Data Science & Artificial Intelligence
Roll No: 24006
