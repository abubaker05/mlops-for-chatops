# MLOps ChatOps Prototype

This is a minimal prototype for a ChatOps-driven MLOps web app. It lets users upload a CSV dataset, train simple models (RandomForest, SVM, Logistic/Linear Regression), evaluate them, and serve predictions.

Structure:
- backend/: Flask API and ML trainer
- frontend/: static HTML/JS prototype
- uploads/: where datasets will be saved (created at runtime)

Quick start (Windows PowerShell):

1. Create a Python venv and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install backend dependencies:

```powershell
cd backend
pip install -r requirements.txt
```

3. Run backend:

```powershell
python app.py
```

4. Open the frontend file `frontend/index.html` in your browser (or serve it with a simple static server).

Notes:
- This is a prototype. For production you should add input validation, authentication, storage (DB), model versioning, and a proper frontend framework.
