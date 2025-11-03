import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import base64
import numpy as np

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading data...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, performs preprocessing for Random Forest classification,
    and returns base64-encoded pickled data with features and target.
    """
    print("Preprocessing data...")
    
    # Decode data
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    print(f"Original shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    print(f"Shape after removing missing values: {df.shape}")
    
    # Create target variable (classification categories)
    # Example: Categorize customers based on BALANCE
    def categorize_balance(balance):
        if balance < 1000:
            return 0  # Low
        elif balance < 3000:
            return 1  # Medium
        else:
            return 2  # High
    
    df['BALANCE_CATEGORY'] = df['BALANCE'].apply(categorize_balance)
    
    print(f"Category distribution:")
    print(df['BALANCE_CATEGORY'].value_counts().sort_index())
    
    # Select features (you can add more features here)
    feature_columns = ["PURCHASES", "CREDIT_LIMIT"]
    X = df[feature_columns]
    y = df["BALANCE_CATEGORY"]
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    
    # Package data
    data_dict = {
        'X': X_scaled,
        'y': y.values,
        'feature_names': feature_columns
    }
    
    serialized_data = pickle.dumps(data_dict)
    return base64.b64encode(serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a Random Forest Classifier model with hyperparameter tuning
    and saves it. Returns training metrics (JSON-serializable).
    """
    print("Building Random Forest model...")
    
    # Decode data
    data_bytes = base64.b64decode(data_b64)
    data_dict = pickle.loads(data_bytes)
    
    X = data_dict['X']
    y = data_dict['y']
    feature_names = data_dict['feature_names']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    
    # Try different n_estimators (number of trees) to find optimal
    results = []
    n_estimators_range = [50, 100, 150, 200, 250]
    
    print("\nTesting different numbers of trees...")
    for n_trees in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'n_estimators': n_trees,
            'accuracy': float(accuracy)
        })
        
        print(f"Trees={n_trees}, Accuracy={accuracy:.4f}")
    
    # Find best n_estimators
    best_result = max(results, key=lambda x: x['accuracy'])
    best_n_estimators = best_result['n_estimators']
    
    print(f"\n{'='*50}")
    print(f"Best number of trees: {best_n_estimators}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    print(f"{'='*50}\n")
    
    # Train final model with best parameters
    print("Training final Random Forest model...")
    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    final_model.fit(X_train, y_train)
    
    # Final evaluation
    y_pred = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, final_model.feature_importances_))
    print(f"\nFeature Importances:")
    for feat, importance in feature_importance.items():
        print(f"  {feat}: {importance:.4f}")
    
    # Save model and test data for evaluation
    model_data = {
        'model': final_model,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'best_n_estimators': best_n_estimators
    }
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")
    
    return {
        'results': results,
        'best_n_estimators': best_n_estimators,
        'best_accuracy': float(best_result['accuracy']),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()}
    }


def load_model_elbow(filename: str, training_results: dict):
    """
    Loads the saved Random Forest model and performs comprehensive evaluation.
    Returns evaluation metrics (JSON-serializable).
    """
    print("Loading Random Forest model...")
    
    # Load model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model_data = pickle.load(open(output_path, "rb"))
    
    model = model_data['model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    feature_names = model_data['feature_names']
    
    print(f"Model loaded from: {output_path}")
    print(f"Number of trees: {model.n_estimators}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"FINAL MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Low (0)', 'Medium (1)', 'High (2)'],
                                zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nConfusion Matrix Explanation:")
    print("Rows = Actual, Columns = Predicted")
    print("Example: cm[0,1] = Low customers predicted as Medium")
    
    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(len(cm)):
        class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Class {i}: {class_acc:.4f}")
    
    # Feature importance
    print("\nFeature Importances:")
    for feat, importance in zip(feature_names, model.feature_importances_):
        print(f"  {feat}: {importance:.4f}")
    
    # Show sample predictions
    print("\nSample Predictions (first 10):")
    for i in range(min(10, len(y_test))):
        print(f"  Actual: {y_test[i]}, Predicted: {y_pred[i]}, " +
              f"Probabilities: {[f'{p:.3f}' for p in y_proba[i]]}")
    
    # Predict on new test data if available
    try:
        test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
        print(f"\nPredicting on test.csv with {len(test_df)} rows...")
        
        test_X = test_df[feature_names]
        test_predictions = model.predict(test_X)
        test_probabilities = model.predict_proba(test_X)
        
        print(f"First prediction: {test_predictions[0]}")
        print(f"First probabilities: {test_probabilities[0]}")
        
        # Return first prediction as int
        first_pred = int(test_predictions[0])
    except Exception as e:
        print(f"Could not predict on test.csv: {e}")
        first_pred = int(y_pred[0])
    
    return {
        'accuracy': float(accuracy),
        'predictions': y_pred[:20].tolist(),
        'probabilities': y_proba[:10].tolist(),
        'feature_importances': model.feature_importances_.tolist(),
        'confusion_matrix': cm.tolist(),
        'first_test_prediction': first_pred,
        'n_estimators': int(model.n_estimators)
    }