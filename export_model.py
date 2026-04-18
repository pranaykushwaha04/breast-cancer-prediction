import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

def main():
    print("Loading data...")
    df = pd.read_csv('data.csv')
    
    # Drop unneeded columns
    if 'Unnamed: 32' in df.columns:
        df.drop('Unnamed: 32', axis=1, inplace=True)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Map diagnosis: M -> 1, B -> 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training CatBoost model...")
    cat_model = CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=0)
    cat_model.fit(X_train_scaled, y_train)
    
    print("Exporting model and scaler...")
    joblib.dump(cat_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Export the feature names so we know exactly what the model expects
    joblib.dump(list(X.columns), 'feature_names.joblib')
    
    print("Export complete! Saved model.joblib, scaler.joblib, and feature_names.joblib.")

if __name__ == "__main__":
    main()
