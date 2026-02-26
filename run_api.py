"""
run_api.py
───────────
Lance l'API FastAPI avec uvicorn.
À utiliser après avoir entraîné le modèle (python train_model.py).

Usage :
    python run_api.py

L'API sera disponible sur :
  → http://localhost:8000
  → http://localhost:8000/docs   (Swagger UI)
  → http://localhost:8000/redoc  (ReDoc)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from config.settings import settings
from src.utils.logger import logger


def main():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║      PME Classifier — Lancement de l'API     ║")
    logger.info(f"║  http://{settings.app_host}:{settings.app_port}                        ║")
    logger.info(f"║  Docs : http://localhost:{settings.app_port}/docs             ║")
    logger.info("╚══════════════════════════════════════════════╝")

    uvicorn.run(
        "src.api.app:app",
        host=str(settings.app_host),
        port=settings.app_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
