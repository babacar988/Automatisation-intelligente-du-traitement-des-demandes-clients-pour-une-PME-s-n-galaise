# 🤖 PME Classifier — Automatisation intelligente des demandes clients

**ISI DITI4 — Cours Machine Learning 2025 — M. Assane BA**

---

## 📁 Architecture du Projet

```
pme_classifier/
│
├── 📄 train_model.py          ← Script d'entraînement (lancer en 1er)
├── 📄 run_api.py              ← Script de lancement de l'API
├── 📄 requirements.txt        ← Dépendances Python
├── 📄 .env                    ← Variables d'environnement (config)
├── 📄 .env.example            ← Template .env public
├── 📄 .gitignore
│
├── 📁 config/
│   ├── __init__.py
│   └── settings.py            ← Configuration centralisée (pydantic-settings)
│
├── 📁 src/
│   ├── __init__.py
│   │
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   └── loader.py          ← Chargement + validation du dataset
│   │
│   ├── 📁 model/
│   │   ├── __init__.py
│   │   ├── trainer.py         ← Entraînement + évaluation du modèle ML
│   │   └── predictor.py       ← Chargement du modèle + inférence
│   │
│   ├── 📁 api/
│   │   ├── __init__.py
│   │   ├── app.py             ← Application FastAPI (lifespan, CORS)
│   │   ├── routes.py          ← Définition des endpoints
│   │   └── schemas.py         ← Modèles Pydantic (validation I/O)
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       └── logger.py          ← Logger centralisé (loguru)
│
├── 📁 data/
│   └── raw/
│       └── dataset.csv        ← Dataset annoté (180 exemples)
│
├── 📁 model/
│   └── saved/
│       └── classifier.joblib  ← Modèle entraîné (généré après train)
│
├── 📁 tests/
│   ├── __init__.py
│   ├── conftest.py            ← Fixtures partagées pytest
│   ├── test_data_loader.py    ← Tests unitaires du DataLoader
│   ├── test_predictor.py      ← Tests unitaires du Predictor
│   └── test_api.py            ← Tests d'intégration de l'API
│
├── 📁 n8n/
│   └── workflow_complet.json  ← Workflow n8n exporté (à importer)
│
└── 📁 logs/
    └── app.log                ← Logs de l'application (généré auto)
```

---

## 🚀 Partie 1 — Python / VSCode

### Prérequis
- Python 3.11+
- pip

### Installation

```bash
# 1. Cloner / ouvrir le dossier dans VSCode
cd pme_classifier

# 2. Créer un environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate
# Activer (Mac/Linux)
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Étape 1 — Entraîner le modèle

```bash
python train_model.py
```

**Ce que fait ce script :**
1. Charge les 180 exemples depuis `data/raw/dataset.csv`
2. Valide les données (colonnes, classes, nulls)
3. Découpe en train (80%) / test (20%) avec stratification
4. Construit le pipeline : `TF-IDF (1-2 grammes)` → `Régression Logistique`
5. Entraîne et évalue (accuracy, F1, matrice de confusion)
6. Lance une validation croisée 5 folds
7. Sauvegarde le modèle dans `model/saved/classifier.joblib`

**Sortie attendue :**
```
╔══════════════════════════════════════════════╗
║   PME Classifier — Entraînement du modèle   ║
╚══════════════════════════════════════════════╝
✓ 180 exemples chargés
✓ Distribution : Information:45 | Commande:45 | Réclamation:45 | Urgence:45
✓ Accuracy : ~85-95%
✓ Modèle sauvegardé : model/saved/classifier.joblib
```

### Étape 2 — Lancer l'API

```bash
python run_api.py
```

L'API est disponible sur :
| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Page d'accueil |
| `http://localhost:8000/docs` | **Documentation Swagger interactive** |
| `http://localhost:8000/redoc` | Documentation ReDoc |
| `http://localhost:8000/health` | État de l'API |
| `http://localhost:8000/classes` | Classes disponibles |
| `POST http://localhost:8000/predict` | **Endpoint de classification** |

### Étape 3 — Tester l'API

**Via Swagger (recommandé pour la démo) :**
Ouvrir `http://localhost:8000/docs` → cliquer `POST /predict` → `Try it out`

**Via curl :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Urgent ! Le paiement par Wave ne fonctionne pas."}'
```

**Réponse attendue :**
```json
{
  "message": "Urgent ! Le paiement par Wave ne fonctionne pas.",
  "classe": "Urgence",
  "confiance": 94.2,
  "action_automatique": "ALERTE PRIORITAIRE : Notification immédiate envoyée au responsable.",
  "reponse_client": "Votre demande urgente a été transmise immédiatement au responsable.",
  "probabilites": {
    "Commande": 1.1,
    "Information": 2.3,
    "Réclamation": 2.4,
    "Urgence": 94.2
  }
}
```

### Étape 4 — Lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ -v --tb=short
```

---

## 🔄 Partie 2 — Workflow n8n

### Prérequis
- n8n installé (`npm install -g n8n`) ou via Docker
- L'API Python doit être en cours d'exécution (`python run_api.py`)
- Un compte Gmail pour les notifications
- Un Google Sheets pour le suivi

### Installation n8n

```bash
# Option A — npm
npm install -g n8n
npx n8n

# Option B — Docker
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

### Importer le workflow

1. Ouvrir `http://localhost:5678`
2. Aller dans **Workflows → Import from File**
3. Sélectionner `n8n/workflow_complet.json`
4. Configurer les credentials (voir ci-dessous)

### Configuration des Credentials

#### Gmail
1. n8n → **Settings → Credentials → Add Credential → Gmail OAuth2**
2. Suivre l'assistant de connexion Google
3. Dans le nœud `7. Envoyer Email`, sélectionner ce credential

#### Google Sheets
1. Créer un Google Sheets avec les onglets :
   - `Urgences` | `Réclamations` | `Commandes` | `Informations`
2. Colonnes à créer (ligne 1) :
   ```
   ID Demande | Date | Canal | Contact Client | Message | Classe ML | Confiance (%) | Action | Numéro Ticket
   ```
3. n8n → **Credentials → Add → Google Sheets OAuth2**
4. Dans le nœud `8. Google Sheets`, remplacer `VOTRE_SPREADSHEET_ID`

### Flux du Workflow

```
[Client envoie un message]
          ↓
[1] Webhook — Réception (POST /webhook/demande-client)
          ↓
[2] Code — Préparation des données (nettoyage, ID unique)
          ↓
[3] HTTP Request → API ML POST /predict
          ↓
[4] Code — Fusion contexte + résultat ML
          ↓
    ┌─────┼──────────┐
    ▼     ▼          ▼
[5a]IF [5b]IF    [5c]IF
Urgence? Réclamation? Commande? → (NON = Information)
    ↓       ↓           ↓              ↓
[6a]Alerte [6b]Ticket [6c]Commande [6d]Info auto
          ↓
[7] Gmail → Email au responsable concerné
          ↓
[8] Google Sheets → Enregistrement dans l'onglet correspondant
          ↓
[9] Respond to Webhook → Réponse automatique au client
```

### Tester le Workflow

```bash
# Envoyer une demande test au webhook n8n
curl -X POST http://localhost:5678/webhook/demande-client \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Urgent ! Le paiement par Wave ne fonctionne pas.",
    "canal": "whatsapp",
    "client_contact": "client@exemple.sn"
  }'
```

---

## 🧠 Choix Techniques

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Vectorisation | TF-IDF (1-2 grammes) | Léger, rapide, efficace sur textes courts |
| Classification | Régression Logistique | Simple, interprétable, retourne des probabilités |
| API | FastAPI | Auto-documentation, validation Pydantic, async |
| Config | pydantic-settings + .env | Paramètres centralisés, pas de hardcoding |
| Logs | loguru | Formatage coloré, rotation automatique des fichiers |
| Tests | pytest + httpx | Standard Python, TestClient FastAPI intégré |
| Automatisation | n8n | Low-code, connecteurs natifs Gmail/Sheets |

---

## 📊 Classes de Classification

| Classe | Exemples | Action n8n |
|--------|----------|------------|
| **Information** | "Avez-vous du riz en stock ?" | Réponse automatique client |
| **Commande** | "Je veux 5 cartons d'eau" | Email stock + Sheets Commandes |
| **Réclamation** | "Produit périmé reçu" | Email service client + ticket + Sheets |
| **Urgence** | "URGENT ! Paiement bloqué" | Email responsable immédiat + Sheets |
#   A u t o m a t i s a t i o n - i n t e l l i g e n t e - d u - t r a i t e m e n t - d e s - d e m a n d e s - c l i e n t s 
 
 