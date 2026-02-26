"""
src/api/routes.py
──────────────────
Définition de tous les endpoints de l'API.

Endpoints :
  GET  /          → Infos générales sur l'API
  GET  /health    → État de santé (modèle chargé ?)
  GET  /classes   → Liste des classes et actions associées
  POST /predict   → Classification d'une demande client
"""

from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import (
    DemandeRequest,
    DemandeResponse,
    HealthResponse,
    ACTIONS,
    REPONSES_AUTO,
)
from src.utils.logger import logger
from config.settings import settings

router = APIRouter()


# ────────────────────────────────────────────────────────────────
# GET /
# ────────────────────────────────────────────────────────────────

@router.get("/", summary="Informations générales")
def root():
    """
    Page d'accueil de l'API.
    Retourne le nom, la version et les endpoints disponibles.
    """
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "description": "API de classification automatique des demandes clients pour PME sénégalaise",
        "endpoints": {
            "GET  /health":  "Vérifier l'état de l'API et du modèle",
            "GET  /classes": "Lister les classes et actions automatiques",
            "POST /predict": "Classifier une demande client",
            "GET  /docs":    "Documentation interactive Swagger (UI)",
            "GET  /redoc":   "Documentation ReDoc",
        },
    }


# ────────────────────────────────────────────────────────────────
# GET /health
# ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="État de l'API")
def health(request: Request):
    """
    Vérifie que l'API est opérationnelle et que le modèle ML est chargé.
    Utilisé par n8n pour vérifier la disponibilité avant d'envoyer des requêtes.
    """
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        modele_charge=predictor.is_loaded,
        classes_disponibles=predictor.classes if predictor.is_loaded else [],
        version=settings.app_version,
    )


# ────────────────────────────────────────────────────────────────
# GET /classes
# ────────────────────────────────────────────────────────────────

@router.get("/classes", summary="Classes et actions disponibles")
def get_classes():
    """
    Retourne les 4 classes de classification et leur action automatique associée.
    Utile pour configurer les branches IF dans le workflow n8n.
    """
    return {
        "classes": list(ACTIONS.keys()),
        "actions": ACTIONS,
        "reponses_auto": REPONSES_AUTO,
    }


# ────────────────────────────────────────────────────────────────
# POST /predict
# ────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=DemandeResponse,
    summary="Classifier une demande client",
    description="""
Analyse le texte d'une demande client et la classe automatiquement dans l'une des 4 catégories :
- **Information** : demande de renseignements
- **Commande** : passage d'une commande
- **Réclamation** : plainte ou insatisfaction
- **Urgence** : problème critique nécessitant une action immédiate

Retourne aussi le niveau de confiance, l'action déclenchée et la réponse à envoyer au client.
    """,
)
def predict(body: DemandeRequest, request: Request):
    """
    Endpoint principal : classifie une demande client.
    Appelé par le nœud HTTP Request dans le workflow n8n.
    """
    predictor = request.app.state.predictor

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Le modèle ML n'est pas disponible. Relancez l'API après l'entraînement.",
        )

    try:
        result = predictor.predict(body.message)
        logger.info(
            f"Prédiction : '{body.message[:50]}...' → {result.classe} "
            f"({result.confiance}%)"
        )

        return DemandeResponse(
            message=body.message,
            classe=result.classe,
            confiance=result.confiance,
            action_automatique=ACTIONS[result.classe],
            reponse_client=REPONSES_AUTO[result.classe],
            probabilites=result.probabilites,
        )

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de la classification : {str(e)}",
        )
