# Breast Cancer Classification using Deep Learning

## 📌 Project Overview
This project focuses on classifying breast cancer histopathology images into **Benign** and **Malignant** categories using deep learning. A pre-trained **ResNet50** Convolutional Neural Network (CNN) with transfer learning was used to build an image classification model.

The project demonstrates how deep learning techniques can support medical image analysis and assist in early breast cancer diagnosis.

---

## 🧠 Problem Statement
Manual analysis of histopathological images is time-consuming and prone to human error. This project aims to automate the classification of breast cancer images using a CNN-based deep learning model to improve accuracy and efficiency.

---

## 📊 Dataset Information
- **Dataset Name:** BreakHis – Breast Cancer Histopathological Images  
- **Source:** Kaggle  
- **Image Type:** Histopathology images  
- **Classes:**  
  - Benign  
  - Malignant  

---

## 🛠 Tools & Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- TensorFlow / Keras  
- Scikit-learn  
- Google Colab  
- Kaggle  

---

## 🔍 Project Workflow
1. Dataset loading and directory structure verification  
2. Image preprocessing and augmentation  
3. Train-validation data generation  
4. Handling class imbalance using class weights  
5. Model building using **ResNet50 (Transfer Learning)**  
6. Model training with callbacks  
7. Fine-tuning selected layers  
8. Model evaluation using accuracy and loss metrics  

---

## 🤖 Model Architecture
- **Base Model:** ResNet50 (pre-trained on ImageNet)  
- **Custom Layers:**  
  - Global Average Pooling  
  - Dense layers  
  - Softmax output layer  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  

---

## 📈 Model Performance
- **Best Validation Accuracy:** ~93%  
- **Final Training Accuracy:** ~93%  

*(Results may vary based on dataset split and training configuration.)*

---

## 📌 Key Learnings
- Implementation of **transfer learning** using ResNet50  
- Handling class imbalance in medical datasets  
- Image preprocessing and data augmentation  
- Training and fine-tuning deep learning models  
- Applying deep learning in healthcare use cases  

---

## 📌 Conclusion
The project shows that deep learning models like ResNet50 can effectively classify breast cancer histopathological images. Transfer learning plays a crucial role in achieving high accuracy even with limited medical data, making it suitable for real-world healthcare applications.

---

## 🚀 Future Improvements
- Experiment with other CNN architectures (EfficientNet, VGG16)  
- Add explainability techniques like **Grad-CAM**  
- Deploy the model using Streamlit or Flask  
- Evaluate performance on unseen datasets  

