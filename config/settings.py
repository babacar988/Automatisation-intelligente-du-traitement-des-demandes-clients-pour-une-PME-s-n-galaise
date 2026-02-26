"""
config/settings.py
──────────────────
Configuration centralisée du projet.
Lit les valeurs depuis le fichier .env via pydantic-settings.
Toute l'application importe Settings() depuis ce module.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# Racine du projet (remonte depuis config/)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Paramètres globaux de l'application.
    Les valeurs sont lues en priorité depuis .env,
    puis depuis les variables d'environnement système.
    """

    # ── Application ────────────────────────────────────────────
    app_name: str = Field(default="PME Classifier API")
    app_version: str = Field(default="1.0.0")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # ── Chemins fichiers ────────────────────────────────────────
    dataset_path: Path = Field(default=BASE_DIR / "data/raw/dataset.csv")
    model_path: Path = Field(default=BASE_DIR / "model/saved/classifier.joblib")
    log_path: Path = Field(default=BASE_DIR / "logs/app.log")

    # ── Hyperparamètres ML ──────────────────────────────────────
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    tfidf_max_features: int = Field(default=5000)
    tfidf_ngram_min: int = Field(default=1)
    tfidf_ngram_max: int = Field(default=2)
    lr_c: float = Field(default=1.0, gt=0)
    lr_max_iter: int = Field(default=1000)

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Instance unique partagée dans tout le projet
settings = Settings()
