import requests
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import json 
from pprint import pprint
import os.path
import seaborn as sns
import matplotlib.pyplot as plt

# URL de l'API ADEME Data Fair (DPE Logements existants depuis 2021)
URL_DPE = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

zone="H1a" #Code du département à analyser

#Nom du fichier qui contiendra les données nettoyées
fname=f'Dpe_{zone}.csv'

#Filtre : 
FILTRE = f'zone_climatique:{zone}'

#initialisation des parametres
PARAMETERS = {
        "q_mode":"simple",
        "size": 1000,
        "qs": FILTRE,
    }

def get_dpe_data(url, params):
    """
    Récupère la première page de données DPE pour Paris (via le code postal 75*).
    """
    #print(f"Tentative de récupération des données DPE pour la commune {Dep}.")
    #print(f"Filtre de recherche (q) utilisé : {params['qs']}")

    #print(url)
    try:
        # Ajout d'un timeout pour éviter les blocages prolongés
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Si la réponse est vide, on récupère un dict vide pour éviter les exceptions
        data = response.json() if response.content else {}
        lines = data.get('results', [])
        total_count = data.get('total', 'N/A')
        # Utiliser .get() pour éviter KeyError si 'next' est absent
        next_rows = data.get('next', None)

        # Débogage (décommenter si nécessaire)
        #print(f"\n Requête réussie.")
        #print(f"Nombre de résultats récupérés dans cette page : {len(lines)}")
        #print(f"Nombre total de résultats disponibles (estimation Data Fair) : {total_count}")
        #print(f"Next URL: {next_rows}")

        return lines, next_rows

    except ValueError as e:
        # Problème lors du décodage JSON
        print(f"Erreur lors du décodage JSON : {e}")
        return [], False

    except requests.exceptions.HTTPError as e:
        # Gère spécifiquement l'erreur 400 ou la fin de la requête forcée
        print(f"Erreur HTTP détectée : {e}")
        # Si c'est une erreur 400, on considère la pagination comme terminée
        if e.response.status_code == 400:
            print("Arrêt suite à l'erreur 400 (Bad Request).")
        return [], False # Retourne une liste vide et False pour l'échec/arrêt

    except requests.exceptions.RequestException as e:
        # Gère les erreurs de connexion, timeout, etc.
        print(f"Erreur lors de la requête HTTP : {e}")
        return [], False # Retourne une liste vide et False pour l'échec

# Chargement des données
existing=False
if os.path.isfile(fname):
    dpe_dep= pd.read_csv(fname)
    existing=True
    print(f"Fichier {fname} existant trouvé avec {len(dpe_dep)} lignes.")
else: 
    all_rows=[]
    
    next_url=URL_DPE
    
    while True :
        
        rows,next_url= get_dpe_data(next_url, PARAMETERS) #rows=les nouvelles lignes, next_url= la nouvelle url
        #print(next_url)
    
        if not next_url:
            # Arrêt si la fonction a rencontré une erreur HTTP (y compris 400)
            break
    
        if not rows:
            # Arrêt si la page est vide (fin naturelle des données)
            print("Aucune ligne trouvée. Fin.")
            break
        
        
        all_rows.extend(rows)
        
        # ✅ MODIFICATION : Afficher le nombre de lignes chargées toutes les 500000 lignes
        if len(all_rows) % 500000 == 0:
            print(f"{len(all_rows)} lignes chargées...")
        
        PARAMETERS=None
          
    print(f"✅ Total final : {len(all_rows)} lignes chargées")
    dpe_H= pd.DataFrame(all_rows)
    dpe_H.to_csv(fname,index=False)
    print(f"Données sauvegardées dans {fname}")
