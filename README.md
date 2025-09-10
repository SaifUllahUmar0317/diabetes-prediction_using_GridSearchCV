# 🩺 Diabetes Prediction App

This project is a **Machine Learning-powered web app** that predicts whether a patient is likely to have diabetes based on medical data.  
It uses **Scikit-Learn** for model training and **Streamlit** for the interactive web interface.  

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
├── diabetes.csv                       
├── model_training.py / .ipynb         
├── diabetes_app.py                    
├── Diabetes predictor model.pkl        
├── Scaler.pkl                          
├── requirements.txt                   
└── README.md                           


---


---

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/SaifUllahUmar0317/diabetes_prediction_using_GridSearchCV.git
cd diabetes_prediction_using_GridSearchCV
pip install -r requirements.txt

---

## ▶️ Usage
Run the Streamlit app with:
```bash
streamlit run user_interface_streamlit_App.py

---

## 📊 Dataset
The dataset used is the **Pima Indians Diabetes Dataset**, which contains medical attributes like:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

**Target column:** `Outcome`  
- `0` → No Diabetes  
- `1` → Diabetes

---

## 🧠 Best Model (via GridSearchCV)
- **Model:** Random Forest Classifier  
- **Best Parameters:**  
  ```json
  { "n_estimators": 100, "max_depth": 10, "min_samples_split": 2 }

---

## 🌐 Web App Preview
The Streamlit app allows users to input patient details and get a diabetes prediction instantly:

- ✅ **No Diabetes** → Green success message  
- 🚨 **Diabetes Detected** → Red warning message

---

## 📦 Requirements
See `requirements.txt`.  
Main libraries used:
- pandas  
- numpy  
- scikit-learn  
- imbalanced-learn  
- streamlit  
- joblib  
- matplotlib  
- seaborn

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---

## 👨‍💻 Author
**Saif Ullah Umar**  
- 💼 Aspiring Machine Learning Engineer  
- 🌍 Based in Pakistan  
- 📧 Email: saifpakistani0317@gmail.com  
- 🔗 GitHub: [SaifUllahUmar0317](https://github.com/SaifUllahUmar0317)  
