import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import pickle

# Data Preparation
data = {
    'Name': ['Saima', 'Laraib', 'Ayesha', 'Fatima', 'Usama'],
    'Marks': [85, 78, 80, 88, 92],
    'CGPA': [3.8, 3.2, 3.5, 3.9, 4.0],
    'Percentage': [90, 82, 85, 92, 95]
}

df = pd.DataFrame(data)

# Preprocessing
scaler = StandardScaler()
df[['Marks', 'CGPA', 'Percentage']] = scaler.fit_transform(df[['Marks', 'CGPA', 'Percentage']])

# Feature Engineering
df['Success'] = df['Percentage'] > 0.85
X = df[['Marks', 'CGPA', 'Percentage']]
y = df['Success']

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file
with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from the file for deployment
with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Flask Application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [data['Marks'], data['CGPA'], data['Percentage']]
    prediction = model.predict([input_data])[0]
    return jsonify({'success': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
