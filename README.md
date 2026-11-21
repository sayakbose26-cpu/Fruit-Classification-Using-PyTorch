# 🍎 Fruit Classification Using PyTorch

A deep learning project that classifies fruit images into five different categories using **PyTorch** and **MobileNetV2**.

The model takes an input image, processes it, and predicts which fruit it belongs to along with its confidence score.

## 🚀 Features

- Image classification using deep learning
- Built with PyTorch and Torchvision
- MobileNetV2 architecture
- Supports five fruit categories
- Displays predicted fruit and confidence score
- Custom image prediction using `predict.py`

## 🍓 Supported Fruit Classes

1. Apple
2. Banana
3. Grape
4. Mango
5. Strawberry

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow

## 📂 Project Structure

```text
Fruit_Classification/
│
├── dataset/
│   ├── Apple/
│   ├── Banana/
│   ├── Grape/
│   ├── Mango/
│   └── Strawberry/
│
├── train.py
├── predict.py
├── fruit_model.pth
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_LINK
```

Navigate to the project folder:

```bash
cd Fruit_Classification
```

Install the required libraries:

```bash
pip install torch torchvision numpy pillow tqdm
```

## ▶️ How to Run

**1. Train the model**

```bash
python train.py
```

**2. Predict a fruit**

```bash
python predict.py
```

Provide an image through the prediction script to classify the fruit.

## 📊 Sample Predictions

```text
================================
       FRUIT CLASSIFICATION
================================
Predicted Fruit: Apple
Confidence: 86.71 %
================================
```

```text
================================
       FRUIT CLASSIFICATION
================================
Predicted Fruit: Banana
Confidence: 97.51 %
================================
```

*These are example prediction scores, not measured test accuracy.*

## 🎯 Project Objective

The main objective of this project is to understand how deep learning and convolutional neural networks can be used for image classification.

This project helped me gain practical experience with:

- Image preprocessing
- Dataset loading
- Model training
- Loss optimization
- Model saving and loading
- Image prediction using PyTorch

## 👨‍💻 Author

**Machine Learning & Deep Learning Enthusiast**

Built as part of my journey into Machine Learning and Deep Learning.
