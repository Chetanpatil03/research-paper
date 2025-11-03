# 🎓 Student Placement Prediction System

## 📘 Overview
The **Student Placement Prediction System** uses Machine Learning to predict whether a student is likely to get placed based on their academic, technical, and personal attributes.  
It helps educational institutions identify students who may need additional training and improve overall placement outcomes.

## 🧠 Key Features
- Predicts student placement status (Placed / Not Placed)  
- Uses multiple ML models (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naïve Bayes)  
- Automatically selects the best-performing model based on accuracy and F1-score  
- Interactive web interface built using **Flask**  
- Visual insights such as feature importance and model performance graphs  

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn  
- **Framework:** Flask  
- **IDE/Platform:** Google Colab, VS Code  
- **Frontend:** HTML, CSS  

## 📂 Project Structure
```

main/
│
├── static/
│   └── style.css             # Frontend styling
│
├── templates/
│   └── index.html            # Web interface for input & prediction
│
├── Model.ipynb               # Jupyter notebook for data analysis and model training
├── app.py                    # Flask application for deployment
├── student_placement_data.csv # Dataset used for model training
└── README.md                 # Project documentation

````

## ⚙️ How It Works
1. **Data Collection:** Dataset of ~450 student records with academic and skill attributes.  
2. **Preprocessing:** Handling missing values, encoding, and scaling.  
3. **Feature Selection:** Important features include CGPA, Technical Skills, Communication Skills, etc.  
4. **Model Training:** Multiple ML models trained; the best one is selected automatically.  
5. **Prediction:** User inputs student details on the Flask app to get placement prediction.  

## 🚀 To Run the Project
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/student-placement-prediction.git
````

2. Navigate to the project folder

   ```bash
   cd student-placement-prediction
   ```
3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application

   ```bash
   python app.py
   ```
5. Open your browser and go to

   ```
   http://127.0.0.1:5000/
   ```

## 📊 Results

* Best Model: Random Forest (Approx. 89% accuracy)
* Key Influencing Features: CGPA, Aptitude Score, Technical & Communication Skills
* Model dynamically selected based on performance metrics

## 🔮 Future Enhancements

* Integration with institutional ERP systems
* Use of Deep Learning for more complex data
* Explainable AI (XAI) to interpret model decisions
* Larger and multi-institution datasets

## 👨‍💻 Author

**Chetan Patil**
BCA Graduate | Data Science Enthusiast | Machine Learning Developer

```

---

Would you like me to make this README **shorter and simpler** (like for a GitHub repo that’s just for personal learning)?
```
