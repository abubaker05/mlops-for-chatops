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
    Trains a model with better error handling for model/task mismatches.
    """
    # Task Type Detection
    if y.dtype == 'object' or (np.issubdtype(y.dtype, np.integer) and y.nunique() < 20):
        is_classification = True
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        is_classification = False
        y = pd.to_numeric(y, errors='coerce').fillna(y.mean())
    
    # <<< --- NEW: ADDED ERROR CHECK FOR MODEL/TASK MISMATCH --- >>>
    if is_classification and model_type == 'linear_regression':
        raise ValueError("Linear Regression is for regression tasks. Please choose a classifier (e.g., Logistic Regression).")
    if not is_classification and model_type == 'logistic_regression':
        raise ValueError("Logistic Regression is for classification tasks. Please choose a regressor (e.g., Linear Regression).")
    # <<< --- END OF NEW CHECK --- >>>

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    for col in list(categorical_features):
        if X_train[col].nunique() > 50:
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)
            categorical_features.remove(col)
    
    if categorical_features:
        X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    
    train_cols = X_train.columns
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    if numeric_features:
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    # Model and parameter grid selection
    if is_classification:
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
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
    if not model_to_tune: raise ValueError(f"Model type '{model_type}' is not defined for this task.")
        
    param_grid = params.get(model_type, {})

    grid_search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, cv=3, scoring=scoring, n_jobs=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    
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