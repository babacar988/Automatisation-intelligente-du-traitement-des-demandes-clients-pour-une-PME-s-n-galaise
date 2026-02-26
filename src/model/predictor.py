"""
src/model/predictor.py
───────────────────────
Chargement du modèle sauvegardé et inférence.

Responsabilités :
  - Charger le modèle depuis le fichier .joblib (une seule fois au démarrage)
  - Exposer une méthode predict() utilisée par l'API
  - Retourner la classe prédite + probabilités par classe
"""

import joblib
import numpy as np
from dataclasses import dataclass

from config.settings import settings
from src.utils.logger import logger


@dataclass
class PredictionResult:
    """
    Résultat structuré d'une prédiction.

    Attributes:
        classe      : classe prédite (ex: "Urgence")
        confiance   : probabilité de la classe prédite en % (ex: 87.3)
        probabilites: dict {classe: probabilité %} pour toutes les classes
    """
    classe: str
    confiance: float
    probabilites: dict[str, float]


class Predictor:
    """
    Charge le modèle ML et effectue des prédictions.
    Conçu pour être instancié une seule fois (singleton via FastAPI lifespan).

    Exemple :
        predictor = Predictor()
        result = predictor.predict("Urgent ! Paiement Wave bloqué.")
        print(result.classe)      # "Urgence"
        print(result.confiance)   # 94.2
    """

    def __init__(self):
        self._pipeline = None

    # ──────────────────────────────────────────────────────────────
    # Chargement
    # ──────────────────────────────────────────────────────────────

    def load(self) -> "Predictor":
        """
        Charge le modèle depuis le disque.
        Doit être appelé avant toute prédiction.

        Raises:
            FileNotFoundError : si le modèle n'existe pas encore
                                (lancer train_model.py d'abord)
        """
        path = settings.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {path}\n"
                "Entraînez d'abord le modèle avec : python train_model.py"
            )

        self._pipeline = joblib.load(path)
        logger.success(f"Modèle chargé : {path}")
        return self

    # ──────────────────────────────────────────────────────────────
    # Prédiction
    # ──────────────────────────────────────────────────────────────

    def predict(self, texte: str) -> PredictionResult:
        """
        Classifie un texte et retourne la prédiction avec les probabilités.

        Args:
            texte : message client à analyser

        Returns:
            PredictionResult avec classe, confiance et probabilités

        Raises:
            RuntimeError : si le modèle n'a pas été chargé
        """
        self._check_loaded()

        # Prédiction
        classe = self._pipeline.predict([texte])[0]
        probas_array = self._pipeline.predict_proba([texte])[0]
        classes = self._pipeline.classes_

        # Construction du dictionnaire de probabilités (arrondi à 1 décimale)
        probabilites = {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(classes, probas_array)
        }

        # Confiance = probabilité de la classe prédite
        confiance = round(float(np.max(probas_array)) * 100, 1)

        return PredictionResult(
            classe=classe,
            confiance=confiance,
            probabilites=probabilites,
        )

    # ──────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def classes(self) -> list[str]:
        """Liste des classes connues par le modèle."""
        self._check_loaded()
        return list(self._pipeline.classes_)

    def _check_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Le modèle n'est pas chargé. Appelez .load() d'abord."
            )
