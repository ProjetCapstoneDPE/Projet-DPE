/**
 * xgboost-predict.js
 * ==================
 * Moteur d'inférence XGBoost côté navigateur.
 * Charge un fichier JSON exporté par export_models_json.py
 * et exécute la prédiction (StandardScaler + OneHotEncoder + XGBoost tree traversal).
 *
 * Usage:
 *   const predictor = new XGBoostPredictor();
 *   await predictor.loadModel('models_json/Bretagne/pipeline_Bretagne_30-50m2.json');
 *   const result = predictor.predict({ surface_habitable_logement: 40, type_batiment: 'maison', ... });
 */

// =============================================
// MAPPINGS QUESTIONNAIRE → FEATURES (porté de app.py)
// =============================================

const DEPT_TO_REGION = {
    // Auvergne-Rhône-Alpes
    '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes',
    '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
    '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
    '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
    // Bourgogne-Franche-Comté
    '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
    '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
    '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
    // Bretagne
    '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
    // Centre-Val de Loire
    '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
    '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
    // Corse
    '2A': 'Corse', '2B': 'Corse',
    // Grand Est
    '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
    '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
    '68': 'Grand Est', '88': 'Grand Est',
    // Hauts-de-France
    '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
    '62': 'Hauts-de-France', '80': 'Hauts-de-France',
    // Île-de-France
    '75': 'Ile-de-France', '77': 'Ile-de-France', '78': 'Ile-de-France',
    '91': 'Ile-de-France', '92': 'Ile-de-France', '93': 'Ile-de-France',
    '94': 'Ile-de-France', '95': 'Ile-de-France',
    // Normandie
    '14': 'Normandie', '27': 'Normandie', '50': 'Normandie',
    '61': 'Normandie', '76': 'Normandie',
    // Nouvelle-Aquitaine
    '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
    '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
    '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
    '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
    // Occitanie
    '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
    '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
    '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie',
    '82': 'Occitanie',
    // Pays de la Loire
    '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
    '72': 'Pays de la Loire', '85': 'Pays de la Loire',
    // Provence-Alpes-Côte d'Azur
    '04': "Provence-Alpes-Côte_d'Azur", '05': "Provence-Alpes-Côte_d'Azur",
    '06': "Provence-Alpes-Côte_d'Azur", '13': "Provence-Alpes-Côte_d'Azur",
    '83': "Provence-Alpes-Côte_d'Azur", '84': "Provence-Alpes-Côte_d'Azur",
    // Outre-Mer
    '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer',
    '974': 'Outre-Mer', '976': 'Outre-Mer'
};

// Mapping dossier → préfixe exact des fichiers de modèles
// (les noms de fichiers peuvent différer des noms de dossiers)
const REGION_TO_FILE_PREFIX = {
    'Auvergne-Rhône-Alpes':       'Auvergne-Rhône-Alpes',
    'Bourgogne-Franche-Comté':    'Bourgogne_Franche_Comte',
    'Bretagne':                   'Bretagne',
    'Centre-Val de Loire':        'Centre-Val de Loire',
    'Corse':                      'Corse',
    'Grand Est':                  'Grand Est',
    'Hauts-de-France':            'Hauts-de-France',
    'Ile-de-France':              'Ile-de-France',
    'Normandie':                  'Normandie',
    'Nouvelle-Aquitaine':         'Nouvelle-Aquitaine',
    'Occitanie':                  'Occitanie',
    'Pays de la Loire':           'Pays_de_la_Loire',
    "Provence-Alpes-Côte_d'Azur": "Provence-Alpes-Côte_d'Azur",
    'Outre-Mer':                  'Outre-Mer'
};

const DEPT_TO_ZONE_CLIMATIQUE = {
    // Hauts-de-France
    '02': 'H1a', '08': 'H1a', '59': 'H1a', '60': 'H1a', '62': 'H1a', '80': 'H1a',
    // Grand Est
    '10': 'H1b', '51': 'H1b', '52': 'H1b', '54': 'H1b', '55': 'H1b',
    '57': 'H1b', '67': 'H1b', '68': 'H1b', '88': 'H1b',
    // Normandie, Bretagne, Pays de la Loire
    '14': 'H2a', '22': 'H2a', '27': 'H2a', '29': 'H2a', '35': 'H2a',
    '44': 'H2a', '49': 'H2a', '50': 'H2a', '53': 'H2a', '56': 'H2a',
    '61': 'H2a', '72': 'H2a', '76': 'H2a', '85': 'H2a',
    // Île-de-France
    '75': 'H1a', '77': 'H1a', '78': 'H1a', '91': 'H1a',
    '92': 'H1a', '93': 'H1a', '94': 'H1a', '95': 'H1a',
    // Occitanie
    '09': 'H3', '11': 'H2c', '12': 'H2c', '30': 'H2c',
    '31': 'H2c', '32': 'H2c', '34': 'H2c', '46': 'H2c',
    '48': 'H2', '65': 'H3', '66': 'H2c', '81': 'H2c', '82': 'H2c',
    // Auvergne-Rhône-Alpes
    '01': 'H1c', '03': 'H1c', '07': 'H2c', '15': 'H1c',
    '26': 'H2b', '38': 'H1c', '42': 'H1c', '43': 'H1c',
    '63': 'H1c', '69': 'H1c', '73': 'H1c', '74': 'H1c',
    // Bourgogne-Franche-Comté
    '21': 'H1c', '25': 'H1c', '39': 'H1c', '58': 'H1b',
    '70': 'H1b', '71': 'H1c', '89': 'H1b', '90': 'H1b',
    // Centre-Val de Loire
    '18': 'H2b', '28': 'H2b', '36': 'H2b', '37': 'H2b', '41': 'H2b', '45': 'H1b',
    // Corse
    '2A': 'H3', '2B': 'H3',
    // Nouvelle-Aquitaine
    '16': 'H2b', '17': 'H2b', '19': 'H1c', '23': 'H1c',
    '24': 'H2c', '33': 'H2c', '40': 'H2c', '47': 'H2c',
    '64': 'H2c', '79': 'H2b', '86': 'H2b', '87': 'H1c',
    // Provence-Alpes-Côte d'Azur
    '04': 'H2c', '05': 'H1c', '06': 'H3', '13': 'H3',
    '83': 'H3', '84': 'H2c',
    // Outre-Mer (tropical climate, using H3 as closest equivalent)
    '971': 'H3', '972': 'H3', '973': 'H3', '974': 'H3', '976': 'H3',
};

function getSurfaceKey(surface) {
    if (typeof surface === 'string') {
        const SURFACE_TO_KEY = {
            'Moins de 30m²': 'inf30m2', '30-50m²': '30-50m2', '50-70m²': '50-70m2',
            '70-90m²': '70-90m2', '90-120m²': '90-120m2', 'Plus de 120m²': 'sup120m2'
        };
        if (SURFACE_TO_KEY[surface]) return SURFACE_TO_KEY[surface];
        surface = parseFloat(surface);
    }
    if (isNaN(surface)) return '70-90m2';

    if (surface < 30) return 'inf30m2';
    if (surface <= 50) return '30-50m2';
    if (surface <= 70) return '50-70m2';
    if (surface <= 90) return '70-90m2';
    if (surface <= 120) return '90-120m2';
    return 'sup120m2';
}

function getSurfaceValue(surface) {
    if (typeof surface === 'string') {
        const SURFACE_TO_VALUE = {
            'Moins de 30m²': 20.0, '30-50m²': 40.0, '50-70m²': 60.0,
            '70-90m²': 80.0, '90-120m²': 105.0, 'Plus de 120m²': 140.0
        };
        if (SURFACE_TO_VALUE[surface]) return SURFACE_TO_VALUE[surface];
    }
    const val = parseFloat(surface);
    return isNaN(val) ? 70.0 : val;
}

const PERIODE_TO_ANNEE = {
    'Avant 1948': 1930.0,
    '1948-1974': 1960.0,
    '1975-1977': 1976.0,
    '1978-1982': 1980.0,
    '1983-1988': 1985.0,
    '1989-2000': 1995.0,
    '2001-2005': 2003.0,
    '2006-2012': 2009.0,
    '2013-2021': 2017.0,
    'Après 2021': 2023.0,
};

const PERIODE_TO_MODEL = {
    'Avant 1948': 'avant 1948',
    '1948-1974': '1948-1974',
    '1975-1977': '1975-1977',
    '1978-1982': '1978-1982',
    '1983-1988': '1983-1988',
    '1989-2000': '1989-2000',
    '2001-2005': '2001-2005',
    '2006-2012': '2006-2012',
    '2013-2021': '2013-2021',
    'Après 2021': 'après 2021',
};

const TYPE_BATIMENT_MAP = {
    'Appartement': 'appartement',
    'Maison': 'maison',
    'Immeuble': 'immeuble',
};

const QUALITE_ISOLATION_MAP = {
    'Très bonne': 'très bonne',
    'Très bonne (triple vitrage)': 'très bonne',
    'Bonne': 'bonne',
    'Bonne (double vitrage récent)': 'bonne',
    'Moyenne': 'moyenne',
    'Moyenne (double vitrage ancien)': 'moyenne',
    'Insuffisante': 'insuffisante',
    'Insuffisante (simple vitrage)': 'insuffisante',
};

const TRAVERSANT_MAP = { 'Oui': 'Oui', 'Non': 'Non' };
const ISOLATION_TOITURE_MAP = { 'Isolée': 'Isolé', 'Non isolée': 'Non Isolé' };

const EMETTEUR_CHAUFFAGE_MAP = {
    // Mappings pour les 6 options du questionnaire vers les 47 options réelles du modèle
    'Radiateur électrique': 'radiateur électrique NFC, NF** et NF***',
    'Convecteur électrique': 'Convecteur électrique NFC, NF** et NF***',
    'Radiateur à eau (chauffage central)': 'Radiateur bitube avec robinet thermostatique sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)',
    'Plancher chauffant': 'Plancher chauffant sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)',
    'Poêle (bois/fioul)': 'Poêle bois',
    'Soufflage air chaud / Climatisation': 'Soufflage d\'air chaud (air soufflé) avec distribution par réseau aéraulique',
};

const INSTALLATION_ECS_MAP = {
    'Individuel (chauffe-eau personnel)': 'individuel',
    'Collectif (chaufferie immeuble)': 'collectif',
    'Mixte': 'mixte (collectif-individuel)',
};

const GENERATEUR_ECS_MAP = {
    'Ballon électrique': 'Ballon électrique à accumulation vertical Catégorie C ou 3 étoiles',
    'Chauffe-eau gaz': 'Chauffe-eau gaz à production instantanée 2001-2015',
    'Chaudière gaz': 'Chaudière gaz standard 2001-2015',
    'Chaudière fioul': 'Chaudière fioul standard 1991-2015',
    'Chauffe-eau thermodynamique': 'CET sur air extérieur après 2014',
    'Pompe à chaleur': 'PAC double service après 2014',
    'Réseau de chaleur': 'Réseau de chaleur isolé',
};

const CLASSE_ALTITUDE_MAP = {
    'Inférieur à 400m': 'inférieur à 400m',
    '400-800m': '400-800m',
    '800-1200m': 'supérieur à 800m',
    'Supérieur à 1200m': 'supérieur à 800m',
};

const HAUTEUR_SOUS_PLAFOND_MAP = {
    'Moins de 2,2m': 2.0,
    '2,2-2,4m': 2.3,
    '2,4-2,6m': 2.5,
    '2,6-3m': 2.8,
    'Plus de 3m': 3.2,
};


/**
 * Convertit les réponses du questionnaire en 22 features du modèle.
 * Port exact de map_responses_to_features() de app.py
 */
function mapResponsesToFeatures(responses, deptCode) {
    const safeGet = (label, def) => responses[label] !== undefined ? responses[label] : def;

    const surfaceVal = safeGet('Surface habitable', 'Je ne sais pas');
    const periodeVal = safeGet('Période de construction', 'Je ne sais pas');
    const typeBatVal = safeGet('Type de bâtiment', 'Je ne sais pas');
    const traversantVal = safeGet('Logement traversant', 'Je ne sais pas');
    const isolToitVal = safeGet('Isolation de la toiture', 'Je ne sais pas');
    const isolEnveloppeVal = safeGet('Qualité isolation globale', 'Je ne sais pas');
    const isolMursVal = safeGet('Isolation des murs', 'Je ne sais pas');
    const isolMenuiseriesVal = safeGet('Qualité des fenêtres', 'Je ne sais pas');
    const isolPlancherVal = safeGet('Isolation du plancher', 'Je ne sais pas');
    const emetteurVal = safeGet('Émetteur de chauffage', 'Je ne sais pas');
    const ecsInstallVal = safeGet('Production eau chaude', 'Je ne sais pas');
    const ecsGenVal = safeGet('Type de chauffe-eau', 'Je ne sais pas');
    const brasseurVal = safeGet('Ventilateur de plafond', 'Je ne sais pas');
    const protSolVal = safeGet('Protection solaire', 'Je ne sais pas');
    const altitudeVal = safeGet('Classe d\'altitude', 'Je ne sais pas');
    const hauteurVal = safeGet('Hauteur sous plafond', 'Je ne sais pas');

    const zoneClim = DEPT_TO_ZONE_CLIMATIQUE[deptCode] || 'H1a';

    return {
        // === CATÉGORIELLES ===
        'type_batiment': TYPE_BATIMENT_MAP[typeBatVal] || 'appartement',
        'zone_climatique': zoneClim,
        'classe_altitude': CLASSE_ALTITUDE_MAP[altitudeVal] || 'inférieur à 400m',
        'chauffage_simplifie': 'Inconnu',
        'logement_traversant_clean': TRAVERSANT_MAP[traversantVal] || 'Inconnu',
        'isolation_toiture_clean': ISOLATION_TOITURE_MAP[isolToitVal] || 'Inconnu',
        'qualite_isolation_enveloppe': QUALITE_ISOLATION_MAP[isolEnveloppeVal] || 'moyenne',
        'periode_construction': PERIODE_TO_MODEL[periodeVal] || '1989-2000',
        'qualite_isolation_murs': QUALITE_ISOLATION_MAP[isolMursVal] || 'moyenne',
        'qualite_isolation_menuiseries': QUALITE_ISOLATION_MAP[isolMenuiseriesVal] || 'moyenne',
        'qualite_isolation_plancher_bas': QUALITE_ISOLATION_MAP[isolPlancherVal] || 'moyenne',
        'type_emetteur_installation_chauffage_n1': EMETTEUR_CHAUFFAGE_MAP[emetteurVal] || 'radiateur électrique NFC, NF** et NF***',
        'type_installation_ecs': INSTALLATION_ECS_MAP[ecsInstallVal] || 'individuel',
        'type_generateur_n1_ecs_n1': GENERATEUR_ECS_MAP[ecsGenVal] || 'Ballon électrique à accumulation vertical Catégorie C ou 3 étoiles',
        'presence_brasseur_air': brasseurVal === 'Oui' ? 1.0 : 0.0,
        'protection_solaire_exterieure': protSolVal === 'Oui' ? 1.0 : 0.0,

        // === NUMÉRIQUES ===
        'surface_habitable_logement': getSurfaceValue(surfaceVal),
        'annee_construction': PERIODE_TO_ANNEE[periodeVal] || 1990.0,
        'hauteur_sous_plafond': HAUTEUR_SOUS_PLAFOND_MAP[hauteurVal] || 2.5,

        // === NUMÉRIQUES ESTIMÉS DYNAMIQUEMENT ===
        // ubat (coefficient de déperdition thermique de l'enveloppe, W/m²·K)
        // Estimé à partir de la qualité d'isolation et de la période de construction
        'ubat_w_par_m2_k': _estimateUbat(isolEnveloppeVal, periodeVal),
        // besoin_chauffage (kWh/an) estimé à partir de ubat, surface, zone climatique
        'besoin_chauffage': _estimateBesoinChauffage(
            _estimateUbat(isolEnveloppeVal, periodeVal),
            getSurfaceValue(surfaceVal),
            zoneClim,
            HAUTEUR_SOUS_PLAFOND_MAP[hauteurVal] || 2.5
        ),
        // apport_solaire (Wh) estimé à partir de la surface et zone climatique
        'apport_solaire_saison_chauffe': _estimateApportSolaire(
            getSurfaceValue(surfaceVal),
            zoneClim
        ),
    };
}

/**
 * Estime le Ubat (W/m²·K) à partir de la qualité d'isolation et de la période.
 * Valeurs typiques DPE :
 *   - RE2020 (après 2021) + très bonne isolation : 0.3-0.5
 *   - RT2012 (2013-2021) + bonne isolation : 0.5-0.8
 *   - Années 2000 + moyenne isolation : 0.8-1.2
 *   - Ancien (avant 1975) + insuffisante : 1.5-2.5
 */
function _estimateUbat(isolEnveloppeVal, periodeVal) {
    // Base par période de construction
    const UBAT_PAR_PERIODE = {
        'Après 2021': 0.35,
        '2013-2021': 0.55,
        '2006-2012': 0.70,
        '2001-2005': 0.85,
        '1989-2000': 1.00,
        '1983-1988': 1.15,
        '1978-1982': 1.30,
        '1975-1977': 1.40,
        '1948-1974': 1.70,
        'Avant 1948': 2.00,
    };

    // Multiplicateur par qualité d'isolation
    const FACTEUR_ISOLATION = {
        'Très bonne': 0.6,
        'Très bonne (triple vitrage)': 0.6,
        'Bonne': 0.8,
        'Bonne (double vitrage récent)': 0.8,
        'Moyenne': 1.0,
        'Moyenne (double vitrage ancien)': 1.0,
        'Insuffisante': 1.3,
        'Insuffisante (simple vitrage)': 1.3,
    };

    const baseUbat = UBAT_PAR_PERIODE[periodeVal] || 1.0;
    const facteur = FACTEUR_ISOLATION[isolEnveloppeVal] || 1.0;

    // Clamp entre 0.25 et 3.0 (plage réaliste DPE)
    return Math.max(0.25, Math.min(3.0, baseUbat * facteur));
}

/**
 * Estime le besoin de chauffage (kWh/an) avec la méthode simplifiée DPE :
 *   besoin ≈ Ubat × surface × hauteur × DJU × 24 / 1000
 * DJU (degrés-jours unifiés) typiques par zone climatique :
 *   H1 : ~2500, H2 : ~2000, H3 : ~1500
 */
function _estimateBesoinChauffage(ubat, surface, zoneClim, hauteur) {
    const DJU_PAR_ZONE = {
        'H1a': 2600, 'H1b': 2700, 'H1c': 2400,
        'H2a': 2100, 'H2b': 2000, 'H2c': 1800, 'H2': 2000,
        'H3': 1400,
    };

    const dju = DJU_PAR_ZONE[zoneClim] || 2200;
    // Formule simplifiée : besoin = Ubat × Sdép × DJU × 24 / 1000
    // Sdép (surface de déperdition) ≈ surface habitable × facteur forme (~2.5 pour appart, ~3.5 pour maison)
    // On simplifie avec un facteur moyen de 2.5
    const surfaceDeperdition = surface * 2.5;
    const besoin = ubat * surfaceDeperdition * dju * 24 / 1000;

    // Clamp entre 500 et 50000 kWh/an
    return Math.max(500, Math.min(50000, Math.round(besoin)));
}

/**
 * Estime l'apport solaire pendant la saison de chauffe (Wh).
 * Dépend de la surface vitrée (~15% surface habitable) et de l'irradiation.
 */
function _estimateApportSolaire(surface, zoneClim) {
    // Irradiation solaire moyenne saison chauffe (Wh/m² de vitrage)
    const IRRADIATION_PAR_ZONE = {
        'H1a': 15000, 'H1b': 16000, 'H1c': 17000,
        'H2a': 17000, 'H2b': 18000, 'H2c': 22000, 'H2': 18000,
        'H3': 25000,
    };

    const irradiation = IRRADIATION_PAR_ZONE[zoneClim] || 18000;
    // Surface vitrée ≈ 15% de la surface habitable
    const surfaceVitree = surface * 0.15;
    // Facteur de transmission solaire moyen des vitrages ≈ 0.5
    const facteurSolaire = 0.5;
    const apportSolaire = surfaceVitree * irradiation * facteurSolaire;

    return Math.round(apportSolaire);
}


// =============================================
// MOTEUR D'INFÉRENCE XGBOOST
// =============================================

class XGBoostPredictor {
    constructor() {
        this.modelData = null;
    }

    /**
     * Charge le fichier modèle.
     * Supporte le chargement via <script> pour éviter les erreurs CORS/file://
     * Remplace automatiquement .json par .js
     * @param {string} url - URL relative du fichier JSON/JS
     */
    async loadModel(url) {
        // Force l'utilisation du fichier .js pour compatibilité file://
        const jsUrl = url.replace('.json', '.js');

        // Extraction du nom du modèle pour récupérer l'objet global
        // ex: "models_json/Bretagne/pipeline_Bretagne_30-50m2.js" -> "pipeline_Bretagne_30-50m2"
        // On utilise decodeURIComponent car les URLs peuvent être encodées (ex: %20 pour espace)
        const filename = decodeURIComponent(jsUrl.split('/').pop().replace('.js', ''));

        console.log(`[XGBoost] Tentative de chargement du modèle: ${filename} via ${jsUrl}`);

        try {
            // Chargement via <script> tag
            await this._loadScript(jsUrl);

            // Récupération depuis l'espace global
            this.modelData = window.XGB_MODELS && window.XGB_MODELS[filename];

            if (!this.modelData) {
                console.error(`[XGBoost] window.XGB_MODELS keys:`, Object.keys(window.XGB_MODELS || {}));
                throw new Error(`Modèle chargé (${jsUrl}) mais introuvable dans window.XGB_MODELS['${filename}']`);
            }

            console.log(`[XGBoost] Modèle chargé avec succès: ${this.modelData.model_name} (${this.modelData.xgboost.learner.gradient_booster.model.trees.length} arbres)`);
        } catch (e) {
            console.error(`[XGBoost] Échec du chargement du modèle: ${jsUrl}`, e);
            throw e;
        }
    }

    /**
     * Helper pour charger un script dynamiquement
     */
    _loadScript(url) {
        return new Promise((resolve, reject) => {
            // Évite de recharger si déjà présent (optionnel mais propre)
            if (document.querySelector(`script[src="${url}"]`)) {
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = url;
            script.onload = () => resolve();
            script.onerror = () => reject(new Error(`Erreur chargement script ${url}`));
            document.head.appendChild(script);
        });
    }

    /**
     * Exécute la pipeline complète : StandardScaler + OneHotEncoder + XGBoost prediction
     * @param {Object} features - Dictionnaire des 22 features { nom: valeur }
     * @returns {number} - La prédiction (consommation annuelle kWh/an)
     */
    predict(features) {
        if (!this.modelData) throw new Error('Aucun modèle chargé');

        const { feature_order, scaler, ohe_categories, xgboost } = this.modelData;

        // 1) StandardScaler sur les colonnes numériques
        const scaledNum = feature_order.numerical.map((col, i) => {
            const val = typeof features[col] === 'number' ? features[col] : parseFloat(features[col]) || 0;
            const res = (val - scaler.mean[i]) / scaler.scale[i];
            console.log(`[XGBoost Scale] ${col}: ${val} -> ${res.toFixed(4)} (mean=${scaler.mean[i].toFixed(2)}, scale=${scaler.scale[i].toFixed(2)})`);
            return res;
        });

        // 2) OneHotEncoder sur les colonnes catégorielles
        const oheVec = [];
        feature_order.categorical.forEach((col, i) => {
            const cats = ohe_categories[String(i)];
            const val = String(features[col] || '');
            // OneHot: un 1.0 si match, sinon NaN (valeur manquante pour XGBoost)
            // C'est CRITIQUE : XGBoost traite les 0 structurels du sparse matrix comme manquants
            cats.forEach(cat => {
                const catNormalized = cat.toLowerCase().trim();
                const valNormalized = val.toLowerCase().trim();
                if (valNormalized === catNormalized) {
                    oheVec.push(1.0);
                    console.log(`[XGBoost OHE] Match: ${col} = ${cat}`);
                } else {
                    oheVec.push(0.0);
                }
            });
        });

        // 3) Vecteur final : [numériques scalées] + [catégorielles one-hot]
        // ATTENTION: l'ordre du ColumnTransformer est num PUIS cat
        const inputVector = [...scaledNum, ...oheVec];

        // 4) XGBoost tree traversal
        return this._predictTrees(inputVector, xgboost);
    }

    /**
     * Traverse tous les arbres XGBoost et somme les prédictions.
     * @param {number[]} inputVector - Vecteur de features après preprocessing
     * @param {Object} xgbJson - Le modèle XGBoost au format JSON natif
     * @returns {number} - La prédiction brute
     */
    _predictTrees(inputVector, xgbJson) {
        const learner = xgbJson.learner;
        const trees = learner.gradient_booster.model.trees;

        // Base score is directly available in learner_model_param
        let baseScore = learner.learner_model_param.base_score;
        if (typeof baseScore === 'string') {
            baseScore = parseFloat(baseScore.replace(/[\[\]]/g, ''));
        }

        let sumTrees = 0;
        for (const tree of trees) {
            sumTrees += this._traverseTree(tree, inputVector);
        }

        const totalPrediction = baseScore + sumTrees;
        
        // Log de diagnostic pour comprendre l'ordre de grandeur
        console.log(`[XGBoost Diagnostic] Base Score: ${baseScore.toFixed(2)}, Somme arbres: ${sumTrees.toFixed(2)}, Total: ${totalPrediction.toFixed(2)}`);

        return Math.max(0, totalPrediction);
    }

    /**
     * Traverse un seul arbre de décision (format plat save_model).
     * @param {Object} tree - Arbre XGBoost au format JSON (arrays)
     * @param {number[]} inputVector - Vecteur de features
     * @returns {number} - La valeur de la feuille atteinte
     */
    _traverseTree(tree, inputVector) {
        const { left_children, right_children, split_indices, split_conditions,
            default_left, base_weights } = tree;

        let nodeIdx = 0;

        // Tant que ce n'est pas une feuille (-1)
        while (left_children[nodeIdx] !== -1) {
            const featureIdx = split_indices[nodeIdx];
            const featureVal = inputVector[featureIdx];

            // Gestion des valeurs manquantes (NaN) via default_left
            if (Number.isNaN(featureVal)) {
                // Si default_left est 1 (vrai), on va à gauche, sinon à droite
                nodeIdx = default_left[nodeIdx] ? left_children[nodeIdx] : right_children[nodeIdx];
            } else if (featureVal < split_conditions[nodeIdx]) {
                nodeIdx = left_children[nodeIdx];
            } else {
                nodeIdx = right_children[nodeIdx];
            }
        }

        return base_weights[nodeIdx];
    }
}


// =============================================
// FONCTION PRINCIPALE DE PRÉDICTION
// =============================================

/**
 * Effectue une prédiction client-side complète.
 * Remplace l'appel à http://localhost:5000/predict
 *
 * @param {Array} responsesArray - Tableau de {label, value} (même format que localStorage['dpe_responses'])
 * @returns {Object} - Résultat au même format que l'API Flask : { prediction, confidence_score, features_used, model_used, region }
 */
async function predictClientSide(responsesArray) {
    // 1) Convertir le tableau [{label, value}] en objet {label: value}
    const responses = {};
    for (const item of responsesArray) {
        responses[item.label] = item.value;
    }

    // 2) Extraire département et surface
    const deptValue = responses['Département'] || '';
    const surfaceValue = responses['Surface habitable'] || '';

    const match = deptValue.match(/^(\d{2,3}[A-B]?)\s*-/);
    const deptCode = match ? match[1] : '75';

    const region = DEPT_TO_REGION[deptCode];
    const surfaceKey = getSurfaceKey(surfaceValue);

    if (!region) throw new Error(`Région non supportée pour le département: ${deptValue}`);
    if (!surfaceKey) throw new Error(`Catégorie de surface non reconnue: ${surfaceValue}`);

    // 3) Mapper les réponses aux 22 features
    const features = mapResponsesToFeatures(responses, deptCode);
    console.log('[XGBoost] Features:', features);

    // 4) Charger les bons modèles JSON
    // Le préfixe des fichiers peut différer du nom de dossier (ex: BFC)
    const filePrefix = REGION_TO_FILE_PREFIX[region] || region;
    const modelDpeUrl  = `models_json/${region}/pipeline_${filePrefix}_${surfaceKey}_dpe.json`;
    const modelElecUrl = `models_json/${region}/pipeline_${filePrefix}_${surfaceKey}_elec.json`;
    console.log(`[XGBoost] Chargement des modèles: ${modelDpeUrl} et ${modelElecUrl}`);

    const predictorDpe = new XGBoostPredictor();
    const predictorElec = new XGBoostPredictor();

    try {
        await predictorDpe.loadModel(modelDpeUrl);
    } catch (e) {
        const fallbackDpeUrl = `models_json/${region}/pipeline_${filePrefix}_dpe.json`;
        console.warn(`[XGBoost] Modèle DPE spécifique introuvable, essai du fallback: ${fallbackDpeUrl}`);
        await predictorDpe.loadModel(fallbackDpeUrl);
    }

    try {
        await predictorElec.loadModel(modelElecUrl);
    } catch (e) {
        const fallbackElecUrl = `models_json/${region}/pipeline_${filePrefix}_elec.json`;
        console.warn(`[XGBoost] Modèle Elec spécifique introuvable, essai du fallback: ${fallbackElecUrl}`);
        await predictorElec.loadModel(fallbackElecUrl);
    }

    // 5) Prédiction Théorique DPE (C_base)
    let predictionTheoretical = predictorDpe.predict(features);

    // 5 bis) Prédiction Théorique Elec
    let predictionTheoreticalElec = predictorElec.predict(features);
    
    // --- SÉCURITÉ : BORNES SUR LA CONSOMMATION THÉORIQUE ---
    const surfaceValForBounds = features.surface_habitable_logement || 70.0;
    const minConso = surfaceValForBounds * 5; 
    const maxConso = surfaceValForBounds * 1200;
    
    if (predictionTheoretical < minConso) {
        console.warn(`[XGBoost] Prédiction DPE très basse (${predictionTheoretical}). Recadrage au min (${minConso}).`);
        predictionTheoretical = minConso;
    } else if (predictionTheoretical > maxConso) {
        console.warn(`[XGBoost] Prédiction DPE très élevée (${predictionTheoretical}). Recadrage au max (${maxConso}).`);
        predictionTheoretical = maxConso;
    }

    if (predictionTheoreticalElec < minConso) {
        console.warn(`[XGBoost] Prédiction Elec très basse (${predictionTheoreticalElec}). Recadrage au min (${minConso}).`);
        predictionTheoreticalElec = minConso;
    } else if (predictionTheoreticalElec > maxConso) {
        console.warn(`[XGBoost] Prédiction Elec très élevée (${predictionTheoreticalElec}). Recadrage au max (${maxConso}).`);
        predictionTheoreticalElec = maxConso;
    }

    console.log(`[XGBoost] Prédiction Théorique DPE: ${predictionTheoretical.toFixed(1)} kWh/an`);
    console.log(`[XGBoost] Prédiction Théorique Elec: ${predictionTheoreticalElec.toFixed(1)} kWh/an`);

    // 6) Consommation Réelle (C_sim_bruit)
    const getVal = (label, def) => {
        const r = responsesArray.find(resp => resp.label === label);
        if (r && r.value !== undefined) {
            const parsed = parseFloat(r.value);
            return isNaN(parsed) ? def : parsed;
        }
        return def;
    };

    const tempValue = getVal('Température de chauffe', 19);
    const nb_occ    = getVal('Nombre d\'occupants', 2);
    const presHours = getVal('Présence par jour', 14);

    const CALIB_DELTA    = 0.02; // Sensibilité température
    const CALIB_OCC      = 250;  // kWh/an par occupant additionnel
    const CALIB_PRESENCE = 0.95; // Facteur d'exposant présence
    const BRUIT_RESIDUEL = 1.00; // Bruit déterministe (moyen)

    const delta_temp = tempValue - 19;
    const presence = presHours / 24.0; 

    // Calcul basé sur la prédiction électrique
    const predictionReal = (
        (0.6 * predictionTheoreticalElec * (1 + CALIB_DELTA * delta_temp))
        + (0.2 * predictionTheoreticalElec)
        + (CALIB_OCC * nb_occ)
    ) * Math.pow(presence, CALIB_PRESENCE) * BRUIT_RESIDUEL;

    console.log(`[XGBoost] Ajustement Réel : C_base_elec=${predictionTheoreticalElec.toFixed(0)}, DeltaT=${delta_temp}°, Occ=${nb_occ}, Pres=${presHours}h => C_réel=${predictionReal.toFixed(0)}`);

    // 7) Score de confiance
    const confidenceScore = Math.min(85, Math.max(40, Math.floor(60 + (predictionTheoretical / 500) * 20)));

    return {
        prediction: predictionTheoretical,     // Énergie Primaire totale (kWh EP/an) - Base du DPE
        predictionElec: predictionTheoreticalElec, // Énergie Finale électricité (kWh EF/an)
        predictionReal: predictionReal,         // Usage réel estimé (kWh EF/an)
        confidence_score: confidenceScore,
        features_used: features,
        model_used: `${region}/${surfaceKey}`,
        model_file_prefix: filePrefix,
        region: region
    };
}


// =============================================
// MOYENNES PAR CLASSE DPE (Référentiel national)
// =============================================

/**
 * Valeurs moyennes typiques des features par classe DPE (A-G).
 * Utilisées pour identifier les points critiques et calculer le potentiel d'amélioration.
 * Sources : données DPE nationales agrégées.
 */
const DPE_CLASS_AVERAGES = {
    'A': {
        ubat_w_par_m2_k: 0.35,
        besoin_chauffage: 2000,
        annee_construction: 2017,
        hauteur_sous_plafond: 2.5,
        qualite_isolation_enveloppe: 'très bonne',
        qualite_isolation_murs: 'très bonne',
        qualite_isolation_menuiseries: 'très bonne',
        qualite_isolation_plancher_bas: 'très bonne',
        isolation_toiture_clean: 'Isolé',
        type_emetteur_installation_chauffage_n1: 'Plancher chauffant sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)',
        type_generateur_n1_ecs_n1: 'CET sur air extérieur après 2014',
        presence_brasseur_air: 1.0,
        protection_solaire_exterieure: 1.0,
    },
    'B': {
        ubat_w_par_m2_k: 0.55,
        besoin_chauffage: 4000,
        annee_construction: 2009,
        hauteur_sous_plafond: 2.5,
        qualite_isolation_enveloppe: 'bonne',
        qualite_isolation_murs: 'bonne',
        qualite_isolation_menuiseries: 'bonne',
        qualite_isolation_plancher_bas: 'bonne',
        isolation_toiture_clean: 'Isolé',
        type_emetteur_installation_chauffage_n1: 'Plancher chauffant sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)',
        type_generateur_n1_ecs_n1: 'CET sur air extérieur après 2014',
        presence_brasseur_air: 1.0,
        protection_solaire_exterieure: 1.0,
    },
    'C': {
        ubat_w_par_m2_k: 0.75,
        besoin_chauffage: 7000,
        annee_construction: 1995,
        hauteur_sous_plafond: 2.5,
        qualite_isolation_enveloppe: 'bonne',
        qualite_isolation_murs: 'bonne',
        qualite_isolation_menuiseries: 'bonne',
        qualite_isolation_plancher_bas: 'moyenne',
        isolation_toiture_clean: 'Isolé',
        type_emetteur_installation_chauffage_n1: 'Radiateur bitube avec robinet thermostatique sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)',
        type_generateur_n1_ecs_n1: 'Chaudière gaz standard 2001-2015',
        presence_brasseur_air: 0.0,
        protection_solaire_exterieure: 1.0,
    },
    'D': {
        ubat_w_par_m2_k: 1.0,
        besoin_chauffage: 11000,
        annee_construction: 1980,
        hauteur_sous_plafond: 2.5,
        qualite_isolation_enveloppe: 'moyenne',
        qualite_isolation_murs: 'moyenne',
        qualite_isolation_menuiseries: 'moyenne',
        qualite_isolation_plancher_bas: 'moyenne',
        isolation_toiture_clean: 'Isolé',
        type_emetteur_installation_chauffage_n1: 'radiateur électrique NFC, NF** et NF***',
        type_generateur_n1_ecs_n1: 'Ballon électrique à accumulation vertical Catégorie C ou 3 étoiles',
        presence_brasseur_air: 0.0,
        protection_solaire_exterieure: 0.0,
    },
    'E': {
        ubat_w_par_m2_k: 1.3,
        besoin_chauffage: 17000,
        annee_construction: 1970,
        hauteur_sous_plafond: 2.8,
        qualite_isolation_enveloppe: 'moyenne',
        qualite_isolation_murs: 'moyenne',
        qualite_isolation_menuiseries: 'insuffisante',
        qualite_isolation_plancher_bas: 'insuffisante',
        isolation_toiture_clean: 'Non Isolé',
        type_emetteur_installation_chauffage_n1: 'Convecteur électrique NFC, NF** et NF***',
        type_generateur_n1_ecs_n1: 'Ballon électrique à accumulation vertical Catégorie C ou 3 étoiles',
        presence_brasseur_air: 0.0,
        protection_solaire_exterieure: 0.0,
    },
    'F': {
        ubat_w_par_m2_k: 1.7,
        besoin_chauffage: 25000,
        annee_construction: 1955,
        hauteur_sous_plafond: 2.8,
        qualite_isolation_enveloppe: 'insuffisante',
        qualite_isolation_murs: 'insuffisante',
        qualite_isolation_menuiseries: 'insuffisante',
        qualite_isolation_plancher_bas: 'insuffisante',
        isolation_toiture_clean: 'Non Isolé',
        type_emetteur_installation_chauffage_n1: 'Convecteur électrique NFC, NF** et NF***',
        type_generateur_n1_ecs_n1: 'Chaudière fioul standard 1991-2015',
        presence_brasseur_air: 0.0,
        protection_solaire_exterieure: 0.0,
    },
    'G': {
        ubat_w_par_m2_k: 2.2,
        besoin_chauffage: 35000,
        annee_construction: 1935,
        hauteur_sous_plafond: 3.0,
        qualite_isolation_enveloppe: 'insuffisante',
        qualite_isolation_murs: 'insuffisante',
        qualite_isolation_menuiseries: 'insuffisante',
        qualite_isolation_plancher_bas: 'insuffisante',
        isolation_toiture_clean: 'Non Isolé',
        type_emetteur_installation_chauffage_n1: 'Convecteur électrique NFC, NF** et NF***',
        type_generateur_n1_ecs_n1: 'Chaudière fioul standard 1991-2015',
        presence_brasseur_air: 0.0,
        protection_solaire_exterieure: 0.0,
    },
};

/**
 * Échelles ordinales pour comparer les features catégorielles.
 * Plus le score est élevé, meilleure est la qualité.
 */
const QUALITY_SCALES = {
    qualite_isolation_enveloppe: {
        'insuffisante': 0, 'moyenne': 1, 'bonne': 2, 'très bonne': 3
    },
    qualite_isolation_murs: {
        'insuffisante': 0, 'moyenne': 1, 'bonne': 2, 'très bonne': 3
    },
    qualite_isolation_menuiseries: {
        'insuffisante': 0, 'moyenne': 1, 'bonne': 2, 'très bonne': 3
    },
    qualite_isolation_plancher_bas: {
        'insuffisante': 0, 'moyenne': 1, 'bonne': 2, 'très bonne': 3
    },
    isolation_toiture_clean: {
        'Non Isolé': 0, 'Inconnu': 0, 'Isolé': 1
    },
};

/**
 * Définition de la direction pour les features numériques.
 * 'lower' = une valeur plus basse est meilleure (ex: ubat, besoin_chauffage, hauteur)
 * 'higher' = une valeur plus haute est meilleure (ex: annee_construction)
 */
const FEATURE_DIRECTION = {
    ubat_w_par_m2_k: 'lower',
    besoin_chauffage: 'lower',
    annee_construction: 'higher',
    hauteur_sous_plafond: 'lower',
    presence_brasseur_air: 'higher',
    protection_solaire_exterieure: 'higher',
};

/**
 * Labels lisibles pour l'affichage des points critiques.
 */
const CRITICAL_POINT_LABELS = {
    ubat_w_par_m2_k: { icon: '🌡️', label: 'Coefficient thermique (Ubat)', unit: 'W/m²·K' },
    besoin_chauffage: { icon: '🔥', label: 'Besoin de chauffage', unit: 'kWh/an' },
    annee_construction: { icon: '📅', label: 'Année de construction', unit: '' },
    hauteur_sous_plafond: { icon: '📏', label: 'Hauteur sous plafond', unit: 'm' },
    qualite_isolation_enveloppe: { icon: '🧱', label: 'Isolation de l\'enveloppe', unit: '' },
    qualite_isolation_murs: { icon: '🧱', label: 'Isolation des murs', unit: '' },
    qualite_isolation_menuiseries: { icon: '🪟', label: 'Qualité des fenêtres', unit: '' },
    qualite_isolation_plancher_bas: { icon: '🧱', label: 'Isolation du plancher', unit: '' },
    isolation_toiture_clean: { icon: '🏚️', label: 'Isolation de la toiture', unit: '' },
    type_emetteur_installation_chauffage_n1: { icon: '🌡️', label: 'Système de chauffage', unit: '' },
    type_generateur_n1_ecs_n1: { icon: '💧', label: 'Production d\'eau chaude', unit: '' },
    presence_brasseur_air: { icon: '💨', label: 'Ventilateur de plafond', unit: '' },
    protection_solaire_exterieure: { icon: '☀️', label: 'Protection solaire', unit: '' },
};

/**
 * Mappings inverses pour l'affichage lisible des valeurs techniques.
 */
const FEATURE_VALUE_LABELS = {
    // Isolation
    'très bonne': 'Très bonne',
    'bonne': 'Bonne',
    'moyenne': 'Moyenne',
    'insuffisante': 'Insuffisante',
    'Isolé': 'Isolé',
    'Non Isolé': 'Non Isolé',
    // Chauffage (Mapping inverse partiel pour la lisibilité)
    'radiateur électrique NFC, NF** et NF***': 'Radiateur électrique (récent)',
    'Convecteur électrique NFC, NF** et NF***': 'Convecteur électrique (ancien)',
    'Radiateur bitube avec robinet thermostatique sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)': 'Radiateur à eau / Chaudière',
    'Plancher chauffant sur réseau individuel eau chaude basse ou moyenne température(inf 65°C)': 'Plancher chauffant',
    'Poêle bois': 'Poêle à bois',
    'Soufflage d\'air chaud (air soufflé) avec distribution par réseau aéraulique': 'Climatisation / Soufflage',
    // ECS
    'Ballon électrique à accumulation vertical Catégorie C ou 3 étoiles': 'Ballon électrique (récent)',
    'Chauffe-eau gaz à production instantanée 2001-2015': 'Chauffe-eau gaz',
    'Chaudière gaz standard 2001-2015': 'Chaudière gaz',
    'Chaudière fioul standard 1991-2015': 'Chaudière fioul',
    'CET sur air extérieur après 2014': 'Chauffe-eau thermodynamique',
    'PAC double service après 2014': 'Pompe à chaleur',
    'Réseau de chaleur isolé': 'Réseau de chaleur',
};

const DPE_GRADES_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];

/**
 * Identifie les points critiques : features de l'utilisateur moins bonnes
 * que la moyenne de la classe DPE juste au-dessus.
 *
 * @param {Object} features - Les 22 features actuelles de l'utilisateur
 * @param {string} currentGrade - La classe DPE actuelle (ex: 'D')
 * @returns {Object} { targetGrade, criticalPoints: [{key, label, icon, unit, currentValue, targetValue, type}] }
 */
function identifyCriticalPoints(features, currentGrade) {
    const gradeIdx = DPE_GRADES_ORDER.indexOf(currentGrade);

    // Si déjà en classe A, pas d'amélioration possible
    if (gradeIdx <= 0) {
        return { targetGrade: 'A', criticalPoints: [] };
    }

    const targetGrade = DPE_GRADES_ORDER[gradeIdx - 1];
    const targetAverages = DPE_CLASS_AVERAGES[targetGrade];
    const criticalPoints = [];

    for (const [key, targetVal] of Object.entries(targetAverages)) {
        const userVal = features[key];
        if (userVal === undefined || userVal === null) continue;

        const meta = CRITICAL_POINT_LABELS[key] || { icon: '📌', label: key, unit: '' };
        let isCritical = false;

        if (QUALITY_SCALES[key]) {
            // Comparaison ordinale (catégorielle)
            const scale = QUALITY_SCALES[key];
            const userScore = scale[userVal] !== undefined ? scale[userVal] : -1;
            const targetScore = scale[targetVal] !== undefined ? scale[targetVal] : -1;
            isCritical = userScore < targetScore;
        } else if (FEATURE_DIRECTION[key]) {
            // Comparaison numérique
            const dir = FEATURE_DIRECTION[key];
            if (dir === 'lower') {
                isCritical = userVal > targetVal; // User a une valeur trop haute
            } else {
                isCritical = userVal < targetVal; // User a une valeur trop basse
            }
        } else {
            // Features catégorielles non-ordinales (chauffage, ECS)
            // On considère critique si la valeur est différente de la cible
            isCritical = String(userVal) !== String(targetVal);
        }

        if (isCritical) {
            criticalPoints.push({
                key,
                label: meta.label,
                icon: meta.icon,
                unit: meta.unit,
                currentValue: userVal,
                targetValue: targetVal,
                type: QUALITY_SCALES[key] ? 'ordinal' : (FEATURE_DIRECTION[key] ? 'numeric' : 'categorical'),
            });
        }
    }

    return { targetGrade, criticalPoints };
}

/**
 * Construit un nouveau jeu de features en remplaçant les points critiques
 * par les valeurs moyennes de la classe DPE cible.
 *
 * Pour ubat et besoin_chauffage, on recalcule les valeurs dérivées
 * cohérentes avec les nouvelles valeurs d'isolation.
 *
 * @param {Object} features - Features actuelles de l'utilisateur
 * @param {Array} criticalPoints - Points critiques identifiés
 * @param {string} targetGrade - Classe DPE cible
 * @returns {Object} - Nouvelles features avec les améliorations appliquées
 */
function buildImprovedFeatures(features, criticalPoints, targetGrade) {
    // Copie profonde des features
    const improved = JSON.parse(JSON.stringify(features));
    const targetAverages = DPE_CLASS_AVERAGES[targetGrade];

    // Appliquer les améliorations
    for (const cp of criticalPoints) {
        improved[cp.key] = cp.targetValue;
    }

    // Recalculer les valeurs dérivées si l'isolation a changé
    // Le ubat et besoin_chauffage doivent être cohérents
    if (improved.ubat_w_par_m2_k !== features.ubat_w_par_m2_k ||
        improved.qualite_isolation_enveloppe !== features.qualite_isolation_enveloppe) {
        // Recalculer besoin_chauffage avec le nouveau ubat
        const zone = improved.zone_climatique || 'H1a';
        const surface = improved.surface_habitable_logement || 70;
        const hauteur = improved.hauteur_sous_plafond || 2.5;
        improved.besoin_chauffage = _estimateBesoinChauffage(
            improved.ubat_w_par_m2_k, surface, zone, hauteur
        );
        // Recalculer apport solaire si pertinent
        improved.apport_solaire_saison_chauffe = _estimateApportSolaire(surface, zone);
    }

    return improved;
}

// Export publics
window.identifyCriticalPoints = identifyCriticalPoints;
window.buildImprovedFeatures = buildImprovedFeatures;
window.FEATURE_VALUE_LABELS = FEATURE_VALUE_LABELS;
