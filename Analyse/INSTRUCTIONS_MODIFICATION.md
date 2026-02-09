# Modifications pour créer des modèles XGBoost par tranche de surface

## 1. Définir les tranches de surface

Ajoutez une cellule après le chargement des données pour définir les tranches :

```python
# Définition des tranches de surface
def categoriser_surface(surface):
    if pd.isna(surface):
        return None
    elif surface <= 50:
        return '0-50m²'
    elif surface <= 100:
        return '50-100m²'
    elif surface <= 150:
        return '100-150m²'
    else:
        return '>150m²'

# Application aux données
df['tranche_surface'] = df['surface_habitable_logement'].apply(categoriser_surface)

# Afficher la distribution
print(df['tranche_surface'].value_counts())
```

## 2. Nettoyage et préparation des données par tranche

Ajoutez après l'analyse des colonnes :

```python
# Filtrer les lignes sans surface
df_clean = df[df['tranche_surface'].notna()].copy()

# Afficher les statistiques par tranche
for tranche in df_clean['tranche_surface'].unique():
    print(f"\n=== Tranche {tranche} ===")
    df_tranche = df_clean[df_clean['tranche_surface'] == tranche]
    print(f"Nombre de logements : {len(df_tranche)}")
    print(f"Surface moyenne : {df_tranche['surface_habitable_logement'].mean():.2f}m²")
```

## 3. Fonction pour créer un modèle XGBoost par tranche

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

def creer_modele_xgboost_par_tranche(df, tranche, features, target='conso_5_usages_e_finale'):
    """
    Crée et entraîne un modèle XGBoost pour une tranche de surface spécifique
    
    Paramètres:
    - df: DataFrame complet
    - tranche: nom de la tranche (ex: '50-100m²')
    - features: liste des colonnes à utiliser comme features
    - target: colonne cible (par défaut 'conso_5_usages_e_finale')
    
    Retourne:
    - model: modèle entraîné
    - metrics: dictionnaire des métriques
    - X_test, y_test: données de test
    """
    
    # Filtrer les données pour cette tranche
    df_tranche = df[df['tranche_surface'] == tranche].copy()
    
    print(f"\n{'='*60}")
    print(f"Entraînement du modèle pour la tranche : {tranche}")
    print(f"{'='*60}")
    print(f"Nombre d'échantillons : {len(df_tranche)}")
    
    # Vérifier qu'il y a assez de données
    if len(df_tranche) < 100:
        print(f"⚠️ Attention : seulement {len(df_tranche)} échantillons pour {tranche}")
        return None, None, None, None
    
    # Préparer X et y
    X = df_tranche[features]
    y = df_tranche[target]
    
    # Supprimer les lignes avec des valeurs manquantes dans la cible
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    # Gérer les valeurs manquantes dans X (remplir avec la médiane)
    X = X.fillna(X.median())
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Créer le modèle XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement
    print("Entraînement en cours...")
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques
    metrics = {
        'tranche': tranche,
        'n_samples': len(df_tranche),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test)
    }
    
    # Affichage des résultats
    print(f"\n📊 Résultats :")
    print(f"  - MAE Train : {metrics['mae_train']:.2f}")
    print(f"  - MAE Test  : {metrics['mae_test']:.2f}")
    print(f"  - R² Train  : {metrics['r2_train']:.4f}")
    print(f"  - R² Test   : {metrics['r2_test']:.4f}")
    
    return model, metrics, X_test, y_test
```

## 4. Boucle pour créer tous les modèles

```python
# Définir les features à utiliser
# (À adapter selon vos colonnes disponibles)
features = [
    'surface_habitable_logement',
    'annee_construction',
    'zone_climatique',
    'type_batiment',
    # ... ajoutez les features pertinentes
]

# Dictionnaires pour stocker les modèles et métriques
modeles = {}
metriques_globales = []

# Créer un modèle pour chaque tranche
for tranche in df_clean['tranche_surface'].unique():
    model, metrics, X_test, y_test = creer_modele_xgboost_par_tranche(
        df_clean, 
        tranche, 
        features
    )
    
    if model is not None:
        modeles[tranche] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }
        metriques_globales.append(metrics)
        
        # Sauvegarder le modèle
        filename = f'modele_xgboost_{tranche.replace("-", "_").replace(">", "plus_")}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Modèle sauvegardé : {filename}\n")

# Afficher un tableau récapitulatif
import pandas as pd
df_metriques = pd.DataFrame(metriques_globales)
print("\n" + "="*80)
print("RÉCAPITULATIF DES MODÈLES PAR TRANCHE")
print("="*80)
print(df_metriques.to_string(index=False))
```

## 5. Visualisation des performances

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Graphique comparatif des performances
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# MAE par tranche
axes[0].bar(df_metriques['tranche'], df_metriques['mae_test'])
axes[0].set_title('MAE par tranche de surface')
axes[0].set_xlabel('Tranche de surface')
axes[0].set_ylabel('MAE (kWh/m²/an)')
axes[0].tick_params(axis='x', rotation=45)

# R² par tranche
axes[1].bar(df_metriques['tranche'], df_metriques['r2_test'])
axes[1].set_title('R² par tranche de surface')
axes[1].set_xlabel('Tranche de surface')
axes[1].set_ylabel('R²')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## 6. Fonction de prédiction

```python
def predire_consommation(surface, autres_features, modeles):
    """
    Prédit la consommation en sélectionnant le modèle approprié selon la surface
    """
    # Déterminer la tranche
    tranche = categoriser_surface(surface)
    
    if tranche not in modeles:
        print(f"❌ Pas de modèle disponible pour la tranche {tranche}")
        return None
    
    # Prédiction avec le bon modèle
    model = modeles[tranche]['model']
    prediction = model.predict([autres_features])
    
    return prediction[0]
```

## Remarques importantes

1. **Ajuster les tranches** : Vous pouvez modifier les seuils selon la distribution de vos données
2. **Sélection des features** : Choisissez les colonnes pertinentes et sans trop de valeurs manquantes
3. **Hyperparamètres** : Vous pouvez optimiser les paramètres XGBoost avec GridSearchCV
4. **Gestion des valeurs manquantes** : Adaptez selon vos besoins (médiane, mode, suppression...)
5. **Encodage** : Les variables catégorielles devront être encodées (One-Hot ou Label Encoding)
