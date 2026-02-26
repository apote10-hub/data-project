import pandas as pd
from sklearn.linear_model import LogisticRegression
import gradio as gr

df = pd.read_csv("heart.csv")

X = df[['age', 'chol', 'thalach']]
y = df['target']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

def predict_risk(age, chol, thalach):
    input_data = pd.DataFrame([[age, chol, thalach]],
                              columns=['age', 'chol', 'thalach'])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return "High risk of heart disease"
    else:
        return "Low risk of heart disease"

app = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Max Heart Rate")
    ],
    outputs="text",
    title="Heart Disease Risk Predictor",
    description="Enter patient details to predict heart disease risk."
)

app.launch()