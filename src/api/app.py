"""
src/api/app.py
───────────────
Point d'entrée de l'application FastAPI.

Utilise le pattern "lifespan" (recommandé FastAPI 0.100+) pour :
  - Charger le modèle ML au démarrage (une seule fois)
  - Libérer les ressources à l'arrêt

Le Predictor est stocké dans app.state pour être accessible
depuis tous les endpoints via request.app.state.predictor.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.model.predictor import Predictor
from src.api.routes import router
from src.utils.logger import logger


# ────────────────────────────────────────────────────────────────
# Lifespan : démarrage / arrêt de l'application
# ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application :
    - Avant yield → code de démarrage (chargement du modèle)
    - Après yield  → code d'arrêt (nettoyage)
    """
    # ── Démarrage ──────────────────────────────────────────────
    logger.info("══════════════════════════════════════════")
    logger.info(f"  Démarrage : {settings.app_name} v{settings.app_version}")
    logger.info("══════════════════════════════════════════")

    predictor = Predictor()
    try:
        predictor.load()
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.warning("L'API démarre en mode dégradé (sans modèle).")

    app.state.predictor = predictor

    yield  # L'application tourne ici

    # ── Arrêt ──────────────────────────────────────────────────
    logger.info("Arrêt de l'application.")


# ────────────────────────────────────────────────────────────────
# Création de l'application
# ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory qui crée et configure l'instance FastAPI."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API de classification automatique des demandes clients pour PME sénégalaise. "
            "Développée avec FastAPI + scikit-learn. "
            "Intégrée dans un workflow n8n pour l'automatisation complète."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS : autorise n8n (localhost:5678) à appeler l'API ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # En production : restreindre à l'URL n8n
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Enregistrement des routes ──────────────────────────────
    app.include_router(router)

    return app


# Instance unique de l'application
app = create_app()
