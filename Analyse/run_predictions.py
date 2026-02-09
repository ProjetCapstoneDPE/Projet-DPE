import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de l'affichage
pd.set_option('display.max_columns', None)

DEPARTMENTS = ['02', '13', '26', '76', '92']

def simplifier_chauffage(val):
    if pd.isna(val):
        return 'Autre/Inconnu'
    val = str(val).lower()
    if 'pac' in val or 'pompe' in val or 'thermodynamique' in val:
        return 'Pompe à Chaleur (PAC)'
    elif 'radiateur' in val or 'convecteur' in val or 'panneau' in val or 'rayonnant' in val or 'standard' in val:
        return 'Radiateur Électrique (Effet Joule)'
    elif 'chaudière' in val:
        return 'Chaudière Électrique'
    else:
        return 'Autre Élec'

def run_analysis_for_dept(dep_code):
    print(f"\n{'='*40}")
    print(f"ANALYSE DU DÉPARTEMENT : {dep_code}")
    print(f"{'='*40}")

    # 1. Chargement des données
    search_path = os.path.join(dep_code, f"*Dpe_dep_*{dep_code}*.csv")
    found_files = glob.glob(search_path)

    if not found_files:
        print(f"ERREUR : Aucun fichier trouvé pour le pattern '{search_path}'.")
        return

    file_path = found_files[0]
    print(f"Chargement du fichier : {file_path}")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return

    # 2. Filtrage et Nettoyage
    if 'type_energie_principale_chauffage' in df.columns:
        df_elec = df[df['type_energie_principale_chauffage'] == 'Électricité'].copy()
    else:
        print("ATTENTION: Colonne 'type_energie_principale_chauffage' absente. Utilisation de tout le dataset.")
        df_elec = df.copy()

    print(f"Logements avec Chauffage Électrique : {df_elec.shape[0]}")

    if df_elec.empty:
        print("Aucune donnée après filtrage.")
        return

    # Création de features
    if 'type_generateur_chauffage_principal' in df_elec.columns:
        df_elec['chauffage_simplifie'] = df_elec['type_generateur_chauffage_principal'].apply(simplifier_chauffage)
    else:
        df_elec['chauffage_simplifie'] = 'Inconnu'

    if 'logement_traversant' in df_elec.columns:
        df_elec['logement_traversant_clean'] = df_elec['logement_traversant'].map({1.0: 'Oui', 0.0: 'Non'}).fillna('Inconnu')
    else:
        df_elec['logement_traversant_clean'] = 'Inconnu'

    if 'isolation_toiture' in df_elec.columns:
        df_elec['isolation_toiture_clean'] = df_elec['isolation_toiture'].map({1.0: 'Isolé', 0.0: 'Non Isolé'}).fillna('Inconnu')
    else:
        df_elec['isolation_toiture_clean'] = 'Inconnu'

    # Sélection des variables
    features_num = [
        'surface_habitable_logement',
        'annee_construction', 
        'hauteur_sous_plafond'
    ]
    features_cat = [
        'type_batiment', 
        'zone_climatique', 
        'classe_altitude',
        'chauffage_simplifie',        
        'logement_traversant_clean',  
        'isolation_toiture_clean'     
    ]
    target_col = 'conso_5_usages_ef'

    # Validation des colonnes
    features_num = [c for c in features_num if c in df_elec.columns]
    features_cat = [c for c in features_cat if c in df_elec.columns]
    
    if target_col not in df_elec.columns:
        print(f"ERREUR: Colonne cible '{target_col}' manquante.")
        return

    # Nettoyage cible
    X = df_elec[features_num + features_cat]
    y = df_elec[target_col]
    
    # Suppression des valeurs aberrantes/nulles sur la cible
    mask = (y > 0) & (y < 100000) # Filtre basique
    X = X[mask]
    y = y[mask]
    
    print(f"Données finales pour entraînement : {X.shape[0]}")
    if X.empty:
        print("Plus de données après nettoyage de la cible.")
        return

    # 3. Pipeline et Modélisation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()) 
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}
    print("\n--- RÉSULTATS DES MODÈLES ---")
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "R2": r2}
        print(f"{name:20} : RMSE = {rmse:.2f} | R² = {r2:.4f}")

    best_model = max(results, key=lambda x: results[x]['R2'])
    print(f"\nMeilleur modèle pour {dep_code} : {best_model} (R² = {results[best_model]['R2']:.4f})")

if __name__ == "__main__":
    for dep in DEPARTMENTS:
        run_analysis_for_dept(dep)
