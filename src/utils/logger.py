"""
src/utils/logger.py
────────────────────
Logger centralisé avec loguru.
Écrit simultanément dans la console (coloré) et dans un fichier rotatif.
Tous les modules importent `logger` depuis ce fichier.

Usage :
    from src.utils.logger import logger
    logger.info("Message d'information")
    logger.warning("Attention !")
    logger.error("Erreur survenue")
"""

import sys
from loguru import logger
from config.settings import settings

# Supprimer le handler par défaut de loguru
logger.remove()

# ── Handler console : couleurs + niveau INFO ────────────────────
logger.add(
    sys.stdout,
    level="DEBUG" if settings.debug else "INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── Handler fichier : rotation 10MB, rétention 7 jours ─────────
settings.log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    settings.log_path,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

__all__ = ["logger"]
