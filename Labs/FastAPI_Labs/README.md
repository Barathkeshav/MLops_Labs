Lab 1 - FastAPI Application

Overview

This lab demonstrates building and running a FastAPI application. The application is a Diabetes Prediction API that uses machine learning to predict diabetes progression based on patient health metrics.

Application Details:

This is a RESTful API for predicting diabetes progression with the following endpoints:

GET / - Health check endpoint

POST /predict - Predict diabetes progression based on patient features

Input Features
The prediction model requires 10 patient health metrics:

age - Age of the patient (normalized)

sex - Sex of the patient (normalized)

bmi - Body mass index (normalized)

bp - Average blood pressure (normalized)

s1 - Total serum cholesterol (normalized)

s2 - Low-density lipoproteins (normalized)

s3 - High-density lipoproteins (normalized)

s4 - Total cholesterol / HDL (normalized)

s5 - Log of serum triglycerides (normalized)

s6 - Blood sugar level (normalized)

