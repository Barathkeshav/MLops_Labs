import pickle
import os
import json
import random
from sklearn.metrics import f1_score, accuracy_score
import joblib
import glob
import sys
import argparse
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Load the trained model
    try:
        model_version = f'model_{timestamp}_dt_model'
        model_path = f'models/{model_version}.joblib'  # Fixed path
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        raise ValueError(f'Failed to load model. File not found: {model_path}')
    except Exception as e:
        raise ValueError(f'Failed to load the latest model: {str(e)}')
    
    # Generate synthetic test data (same as training)
    try:
        X, y = make_classification(
            n_samples=random.randint(100, 2000),  # Changed from 0 to 100
            n_features=6,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            random_state=0,
            shuffle=True,
        )
        print(f"Test data generated: {X.shape[0]} samples")
    except Exception as e:
        raise ValueError(f'Failed to generate test data: {str(e)}')
    
    # Make predictions and calculate metrics
    y_predict = model.predict(X)
    
    metrics = {
        "F1_Score": f1_score(y, y_predict),
        "Accuracy": accuracy_score(y, y_predict),
        "num_samples": X.shape[0]
    }
    
    print(f"Evaluation Metrics:")
    print(f"  F1 Score: {metrics['F1_Score']:.4f}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    
    # Create metrics directory if it doesn't exist
    if not os.path.exists('metrics/'): 
        os.makedirs("metrics/")
    
    # Save metrics to a JSON file with correct path
    metrics_filename = f'metrics/{timestamp}_metrics.json'  # Fixed path
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    print(f"Metrics saved successfully to: {metrics_filename}")