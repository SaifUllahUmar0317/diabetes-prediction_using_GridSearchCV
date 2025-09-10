# 🩺 Diabetes Prediction App (Improved Version)

This project is an **improved version** of my earlier Diabetes Prediction project.  
Unlike the previous version, this one includes **hyperparameter tuning, advanced evaluation metrics, and a Streamlit-based interactive web app** for real-time predictions.  

It predicts whether a patient is likely to have diabetes based on medical data using **Machine Learning**.  
The backend uses **Scikit-Learn**, while the frontend is powered by **Streamlit**.   

---

## 🚀 Features
- Trains multiple ML models (Decision Tree, KNN, Random Forest, SVM)  
- Performs **hyperparameter tuning** using GridSearchCV  
- Handles **imbalanced data** with SMOTE  
- Scales features with **MinMaxScaler**  
- Provides **metrics evaluation** (accuracy, precision, recall, F1-score)  
- Saves the trained model and scaler (`.pkl` files)  
- Streamlit app for user-friendly predictions  

---

## 📂 Project Structure
```text
├── diabetes.csv                       
├── main.py / .ipynb         
├── user_interface.py                    
├── Diabetes predictor model.pkl        
├── Scaler.pkl                          
├── requirements.txt                   
└── README.md                           
```

---

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/SaifUllahUmar0317/diabetes_prediction_using_GridSearchCV.git
cd diabetes_prediction_using_GridSearchCV
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the **Streamlit app**:
```bash
streamlit run diabetes_app.py
```

This will start a local web server, and you can access the app in your browser.  

---

## 📊 Dataset
The dataset used is **Pima Indians Diabetes Dataset** (from Kaggle / UCI repository).  
It contains medical diagnostic measurements of patients (e.g., glucose level, BMI, age) and whether they have diabetes.  

---

## 🧠 Model Training
- Data preprocessing (handling imbalance with **SMOTE**)  
- Feature scaling (**MinMaxScaler**)  
- Models trained: **Decision Tree, KNN, Random Forest, SVM**  
- Hyperparameter tuning using **GridSearchCV**  

---

## 🏆 Best Model
The **best performing model** was selected based on accuracy, precision, recall, and F1-score.  
The trained model is saved as `Diabetes predictor model.pkl` and can be loaded for predictions.  

---

## 👤 Author
**Saif Ullah Umar**  
📧 Email: saifpakistani0317@gmail.com  
🌐 GitHub: [SaifUllahUmar0317](https://github.com/SaifUllahUmar0317)  

---

## 📜 License
This project is licensed under the **MIT License** - feel free to use, modify, and share.
