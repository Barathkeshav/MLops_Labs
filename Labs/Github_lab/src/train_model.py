# from sklearn.datasets import fetch_rcv1
import mlflow
import datetime
import os
import pickle
import random
# import sklearn
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Generate synthetic data for demonstration
    X, y = make_classification(
        n_samples=random.randint(100, 2000),  # Changed from 0 to 100 to avoid empty dataset
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=True,
    )
    
    # Save the generated data
    if not os.path.exists('data'): 
        os.makedirs('data/')
    
    with open('data/data.pickle', 'wb') as data_file:
        pickle.dump(X, data_file)
        
    with open('data/target.pickle', 'wb') as target_file:
        pickle.dump(y, target_file)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Reuters Corpus Volume"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")
    
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_name}"):
        
        # Log parameters
        params = {
            "dataset_name": dataset_name,
            "number of datapoint": X.shape[0],
            "number of dimensions": X.shape[1]
        }
        
        mlflow.log_params(params)
        
        # Train the Random Forest model
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X, y)
        
        # Make predictions and evaluate
        y_predict = forest.predict(X)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(y, y_predict),
            'F1 Score': f1_score(y, y_predict)
        })
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models/'): 
            os.makedirs("models/")
        
        # Save the trained model with timestamp versioning
        model_version = f'model_{timestamp}'
        model_filename = f'models/{model_version}_dt_model.joblib'  # Fixed path
        dump(forest, model_filename)
        
        print(f"Model saved successfully as: {model_filename}")
        print(f"Accuracy: {accuracy_score(y, y_predict):.4f}")
        print(f"F1 Score: {f1_score(y, y_predict):.4f}")