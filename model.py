import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_cancer_data

def train_and_evaluate_model():
    """
    Loads data, preprocesses it, trains a Random Forest model,
    and returns the trained model, scaler, and accuracy score.
    """
    # 1. Fetch data from our loader
    df = load_cancer_data()
    
    # 2. Clean data
    # Drop columns that aren't features (id and Unnamed: 32 which is often an artifact)
    columns_to_drop = ['id']
    if 'Unnamed: 32' in df.columns:
        columns_to_drop.append('Unnamed: 32')
    df = df.drop(columns=columns_to_drop)
    
    # Encode target: Malignant (M) = 1, Benign (B) = 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Split features (X) and target (y)
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    # For user ease in the UI, let's keep track of core features 
    # (Using a core subset keeps the UI clean instead of 30 inputs)
    core_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
    ]
    
    # Filter X for the UI demonstration to make it user friendly, or use all.
    # We will use all 30 features here for optimal model accuracy.
    feature_names = X.columns.tolist()
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Scaling (Important for medical data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, feature_names

if __name__ == "__main__":
    # Test training locally
    print("Training model...")
    model, scaler, accuracy, features = train_and_evaluate_model()
    print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")