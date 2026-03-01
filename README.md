🫀 Heart Disease Risk Prediction System

A production-oriented Machine Learning web application that predicts the probability of heart disease using a Random Forest classifier and generates a structured medical-style report with explainable insights.

📌 Problem Statement

Early detection of cardiovascular disease significantly reduces mortality risk. This project aims to build a reliable and interpretable ML-based prediction system that:
	•	Predicts heart disease probability
	•	Provides confidence score (risk percentage)
	•	Generates downloadable structured reports
	•	Offers recommendation logic based on predicted risk

🧠 Model Architecture
🔹 Algorithm Used
	•	Random Forest Classifier

Why Random Forest?
	•	Handles non-linearity effectively
	•	Robust to outliers
	•	Reduces overfitting via ensemble learning
	•	Provides feature importance for interpretability

📊 Model Performance
Confusion Matrix:
[[70 12]
 [ 9 93]]

Evaluation Metrics:
Metric	Score
Accuracy	88.6%
Precision (Class 1)	0.89
Recall (Class 1)	0.91
F1-Score	0.90
ROC-AUC	0.93
CV Mean Score	0.82

Interpretation:
	•	High recall (0.91) ensures fewer false negatives (critical in medical use cases).
	•	ROC-AUC of 0.93 indicates strong class separability.
	•	Balanced precision-recall shows stable model generalization.

🔄 ML Pipeline
	1.	Data Cleaning & Preprocessing
	•	Handling missing values
	•	Feature scaling
	•	Encoding categorical features
  
	2.	Train-Test Split
	•	Stratified splitting
  
	3.	Model Training
	•	RandomForestClassifier
	•	Hyperparameter tuning
  
	4.	Evaluation
	•	Accuracy
	•	Precision / Recall / F1
	•	ROC-AUC
	•	Cross-validation
  
	5.	Model Serialization
	•	Saved using joblib for production use


🚀 Application Features
1️⃣ Probability-Based Prediction

Instead of binary output, the system returns:
Heart Disease Risk: 78%
Risk Category: High Risk

Uses predict_proba() to extract probability for positive class.

2️⃣ Explainability
	•	Displays top contributing features using Random Forest feature importance.
	•	Helps users understand prediction reasoning.

3️⃣ Dynamic Report Generation

After prediction, users can download a structured PDF report containing:
	•	Patient clinical data
	•	Predicted probability (%)
	•	Risk classification
	•	Model used (Random Forest)
	•	Timestamp
	•	Recommendation engine output
	•	QR code for report verification

PDF generated using:
	•	ReportLab
	•	QR Code integration

4️⃣ Recommendation Engine
Rule-based recommendation layer based on probability thresholds:

Risk Range	Recommendation
0–30%	Preventive lifestyle maintenance
31–60%	Regular monitoring advised
61–100%	Immediate cardiology consultation

🛠️ Tech Stack
Backend:
	•	Python
	•	Flask

Machine Learning:
	•	Scikit-learn
	•	RandomForestClassifier
	•	NumPy
	•	Pandas

Visualization:
	•	Matplotlib
  .Seaborn

Report Generation:
	•	ReportLab
	•	QRCode

Frontend:
	•	HTML
	•	CSS

📁 Project Structure
├── app.py
├── models/
│   ├── heart_model_advanced.pkl
│   └── scaler_advanced.pkl
├── templates/
├── static/
├── reports/
└── README.md


🧪 Key Technical Highlights
	•	Ensemble-based ML model
	•	Probability-driven prediction
	•	ROC-AUC optimization
	•	Cross-validation integrated
	•	Production-ready model serialization
	•	Structured PDF generation
	•	Explainable AI component
	•	Modular Flask architecture


🔮 Future Enhancements
	•	SHAP-based explainability
	•	Docker containerization
	•	Cloud deployment (Render / AWS)
	•	REST API version
	•	Real-time health monitoring integration

👨‍💻 Author
Mohammad Akif Akhtar
AI/ML Engineer

