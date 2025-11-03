Lab 4 - Github Lab

This project demonstrates an automated machine learning pipeline using GitHub Actions for continuous model training, evaluation, and versioning. The workflow automatically trains a Random Forest classifier on synthetic data, evaluates its performance, and versions both the model and metrics with timestamps whenever changes are pushed to the main branch.
Key Features:

Automated Training Pipeline: Triggers automatically on push to main branch or on a daily schedule
Model Versioning: Each trained model is saved with a unique timestamp identifier for easy tracking and comparison
Performance Metrics: Automatically calculates and stores F1 Score and Accuracy for each model version
Data Persistence: Saves training data, models, and evaluation metrics to dedicated directories
GitHub Actions Integration: Leverages CI/CD best practices with automated workflows that handle the complete ML lifecycle

Workflow Components:
The pipeline consists of automated steps that set up the Python environment, install dependencies, generate synthetic classification data, train a Random Forest model, evaluate performance metrics, version the trained model with timestamps, and commit all artifacts back to the repository. Two workflows are configured: one that runs on every push to main for immediate retraining, and a periodic workflow scheduled to run daily for regular model updates.

Changes Made

Model_train.py:
The training script has been updated with several critical improvements to enhance reliability and functionality. The most significant fix addresses the model file path issue—models are now correctly saved to the models/ directory instead of the root directory. Additionally, the synthetic data generation now uses a minimum of 100 samples (previously 0) to prevent empty dataset errors that could cause training failures. The data directory creation logic has been streamlined for better efficiency, and informative print statements have been added to provide clear feedback on model performance metrics (Accuracy and F1 Score) and successful file saves. These changes ensure the GitHub Actions workflow runs smoothly and models are properly versioned and stored in the expected location.

Changes were also made to evaluate_model.py according to changes to model_train.py

The workflows also have been moved outside the labs folder and the yml files have some changes