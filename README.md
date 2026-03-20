# Breast Cancer Detection using Deep Learning

## Overview
This project is a Final Year Project (FYP) focused on detecting breast cancer from mammography images using Deep Learning techniques. The system uses a Convolutional Neural Network (CNN) to classify medical images into two categories, either as benign and malignant.

The objective of this project is to assist in early detection of breast cancer, improve diagnostic efficiency and support medical professionals in decision-making.

---

## Dataset

The dataset used in this project is the CBIS-DDSM (Curated Breast Imaging Subset of DDSM).

CBIS-DDSM is an updated and standardized version of the original DDSM dataset and is widely used in medical imaging research. It contains:

- High-quality digitized mammography images  
- Craniocaudal (CC) and mediolateral oblique (MLO) views  
- Annotated lesions with ROI masks  
- Labels indicating benign or malignant cases  
- Standardized preprocessing and organization  

Using this dataset ensures reproducibility and allows comparison with other research works in the field.

Dataset was accessed and downloaded from Kaggle using the Kaggle API:

https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

A Kaggle API token was generated and used to download the dataset programmatically.

---

## Technologies Used

- Python: https://www.python.org/  
- TensorFlow: https://www.tensorflow.org/  
- Keras: https://keras.io/  
- NumPy: https://numpy.org/  
- OpenCV: https://opencv.org/  
- Scikit-learn: https://scikit-learn.org/  
- Matplotlib: https://matplotlib.org/  

---

## Model Architecture

- Convolutional Neural Network (CNN)  
- Image preprocessing including resizing and normalization  
- Feature extraction using convolutional layers  
- Pooling layers for dimensionality reduction  
- Fully connected dense layers for classification  
- Binary classification output:
  - Benign  
  - Malignant  

---

## Project Structure

Breast-Cancer-Detection/
│
├── ML-Pipeline/          Model training, preprocessing, and evaluation scripts  
├── Backend/              Backend logic (if applicable)  
├── Frontend/             User interface (if applicable)  
├── Final Report.pdf      Project documentation/report  
├── requirements.txt      Project dependencies  
├── .gitignore  
└── README.md  

---

## Installation and Setup

### 1. Clone the repository
git clone https://github.com/talhahashmii/Breast-Cancer-Detection.git

### 2. Navigate to the project directory
cd Breast-Cancer-Detection

### 3. Create a virtual environment
python -m venv venv

Activate the virtual environment:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

---

## How to Run the Project

Navigate to the ML-Pipeline folder:

cd ML-Pipeline

Run the preprocessing script if available.

Train the model:
python train.py

Evaluate or test the model:
python test.py

(Adjust the commands based on actual script filenames in the project.)

---

## Results

The trained CNN model is evaluated using standard classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

These metrics are used to assess the performance of the model in classifying mammography images.

---

## Future Improvements

- Improve model performance using advanced architectures such as ResNet or VGG  
- Apply transfer learning for better feature extraction  
- Develop a web-based interface for real-time predictions  
- Enhance preprocessing and data augmentation techniques  
- Deploy the model as an API or full-stack application  

---

## Repository

https://github.com/talhahashmii/Breast-Cancer-Detection

---

## Author

Talha Hashmi  
Bachelor’s in Computer Science  
Goldsmiths, University of London (via Beaconhouse International College)

---

## Acknowledgements

- Kaggle for providing access to the CBIS-DDSM dataset  
- TensorFlow and Keras communities for deep learning frameworks  
- Open-source contributors for tools and libraries used in this project  

---
