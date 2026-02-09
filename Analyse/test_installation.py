"""
Script de vérification de l'installation de XGBoost
"""

print("Vérification des installations...")
print("-" * 50)

try:
    import xgboost as xgb
    print(f"✓ XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"✗ XGBoost: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

try:
    import seaborn as sns
    print(f"✓ Seaborn version: {sns.__version__}")
except ImportError as e:
    print(f"✗ Seaborn: {e}")

print("-" * 50)
print("Toutes les dépendances sont installées ! 🎉")
