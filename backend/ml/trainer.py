import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def train_model(X, y, model_type='decision_tree'):
    """
    Trains a model with robust preprocessing, missing value imputation, and 
    stable cross-validation for high reliability across various datasets.
    """
    
    # 1. Task Type Detection
    if y.dtype == 'object' or (np.issubdtype(y.dtype, np.integer) and y.nunique() < 20):
        is_classification = True
        # Handle case where target column is entirely missing
        if y.isnull().all():
            raise ValueError("Target column contains all missing values and cannot be used.")
            
        le = LabelEncoder()
        # Ensure LabelEncoder only fits on non-NaN values if any managed to slip through
        y = le.fit_transform(y.dropna()) 
    else:
        is_classification = False
        # Impute missing Y values with the mean for regression stability
        y = pd.to_numeric(y, errors='coerce').fillna(y.mean())
    
    # Check for empty feature matrix after target selection/dropping
    if X.empty or len(X.columns) == 0:
        raise ValueError("Feature matrix (X) is empty after target column removal.")
    
    # 2. Model/Task Mismatch Check (Original Logic - Good)
    if is_classification and model_type == 'linear_regression':
        raise ValueError("Linear Regression is for regression tasks. Please choose a classifier (e.g., Logistic Regression).")
    if not is_classification and model_type == 'logistic_regression':
        raise ValueError("Logistic Regression is for classification tasks. Please choose a regressor (e.g., Linear Regression).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ==================================================================================
    # 🚨 IMPROVEMENT 1: Robust Missing Value Imputation (Solves NaN errors)
    # Impute missing values on X_train and apply to X_test immediately after the split.
    # ==================================================================================
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if X_train[col].dtype == 'object':
                # Impute categorical with the mode (most frequent)
                mode_value = X_train[col].mode()[0] if not X_train[col].mode().empty else 'MISSING'
                X_train[col].fillna(mode_value, inplace=True)
                X_test[col].fillna(mode_value, inplace=True)
            else:
                # Impute numeric with the mean
                mean_value = X_train[col].mean()
                X_train[col].fillna(mean_value, inplace=True)
                X_test[col].fillna(mean_value, inplace=True)
    # ==================================================================================

    # 3. Preprocessing Setup
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Drop high-cardinality categorical features (Original Logic - Good)
    features_to_remove = []
    for col in list(categorical_features):
        # IMPROVEMENT 2: Skip dropping if the column is too small to split
        if X_train[col].nunique() > 50 and X_train.shape[0] > 50:
            features_to_remove.append(col)
        
    X_train.drop(columns=features_to_remove, inplace=True, errors='ignore')
    X_test.drop(columns=features_to_remove, inplace=True, errors='ignore')
    categorical_features = [f for f in categorical_features if f not in features_to_remove]
    numeric_features = [f for f in numeric_features if f not in features_to_remove]

    # One-Hot Encoding
    if categorical_features:
        X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    
    # Align columns after encoding
    train_cols = X_train.columns
    # IMPROVEMENT 3: Handle empty dataframes after processing (e.g. if only the target remains)
    if train_cols.empty:
        raise ValueError("All input features were dropped during preprocessing (e.g., high cardinality). Cannot train.")
        
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    # Scaling Numeric Features
    if numeric_features:
        scaler = StandardScaler()
        # Select only the columns that are still present (not dropped by high cardinality)
        cols_to_scale = [col for col in numeric_features if col in X_train.columns]
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 4. Model Setup
    # Model and parameter grid selection (Unchanged)
    if is_classification:
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, solver='liblinear'), # safer solver for small datasets
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'lightgbm': LGBMClassifier(random_state=42)
        }
        params = {
            'logistic_regression': {'C': [0.1, 1.0]},
            'decision_tree': {'max_depth': [5, 10, 20]},
            'svm': {'C': [0.1, 1.0]},
            'lightgbm': {'n_estimators': [50], 'learning_rate': [0.1]}
        }
        scoring = 'accuracy'
    else: # Regression
        models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'svm': SVR(),
            'lightgbm': LGBMRegressor(random_state=42)
        }
        params = {
            'linear_regression': {},
            'decision_tree': {'max_depth': [5, 10, 20]},
            'svm': {'C': [0.1, 1.0]},
            'lightgbm': {'n_estimators': [50], 'learning_rate': [0.1]}
        }
        scoring = 'r2'

    model_to_tune = models.get(model_type)
    if not model_to_tune: 
        raise ValueError(f"Model type '{model_type}' is not defined for this task.")
        
    param_grid = params.get(model_type, {})

    # ==================================================================================
    # 🚨 IMPROVEMENT 4: Robust Cross-Validation (Solves n_splits error)
    # Dynamically set CV splits based on the task and data size.
    # ==================================================================================
    n_samples = X_train.shape[0]
    if is_classification:
        # Use cv=2 for classification (safest) to avoid "n_splits" error on imbalanced/small classes
        cv_splits = 2 
    elif n_samples < 5:
        # If dataset is tiny, use no cross-validation
        cv_splits = 2 # minimum cv is 2 for GridSearchCV 
    else:
        # Use 3 splits for general regression tasks
        cv_splits = 3 
        
    # Set n_jobs to -1 to use all cores if performance is an issue, but keeping it at 1 for initial stability
    grid_search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, cv=cv_splits, scoring=scoring, n_jobs=1)
    
    # Final check before fit
    if X_train.shape[0] < cv_splits:
        raise ValueError(f"Not enough data ({X_train.shape[0]} samples) for {cv_splits}-fold cross-validation. Try a larger dataset.")
    
    grid_search.fit(X_train, y_train)
    # ==================================================================================
    
    best_model = grid_search.best_estimator_
    
    # Handle the case where predict can't run on the (potentially empty) X_test
    try:
        preds = best_model.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Prediction failed on test set. Data issue? Error: {str(e)}")
    
    # 5. Metrics Calculation (Unchanged)
    metrics = {}
    if is_classification:
        metrics['accuracy'] = round(accuracy_score(y_test, preds), 4)
        metrics['precision'] = round(precision_score(y_test, preds, average='weighted', zero_division=0), 4)
        metrics['recall'] = round(recall_score(y_test, preds, average='weighted', zero_division=0), 4)
        metrics['f1_score'] = round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)
    else:
        metrics['mse'] = round(mean_squared_error(y_test, preds), 4)
        metrics['r2_score'] = round(r2_score(y_test, preds), 4)

    return best_model, metrics
