"""
train_model.py
───────────────
Script principal d'entraînement du modèle ML.
À lancer une seule fois avant de démarrer l'API.

Usage :
    python train_model.py

Ce script :
  1. Charge le dataset depuis data/raw/dataset.csv
  2. Valide les données
  3. Entraîne le pipeline TF-IDF → Régression Logistique
  4. Évalue les performances (accuracy, F1, matrice de confusion)
  5. Sauvegarde le modèle dans model/saved/classifier.joblib
"""

import sys
from pathlib import Path

# S'assurer que la racine du projet est dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.model.trainer import ModelTrainer
from src.utils.logger import logger


def main():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   PME Classifier — Entraînement du modèle   ║")
    logger.info("╚══════════════════════════════════════════════╝")

    try:
        # ── Étape 1 : Chargement des données ──────────────────
        loader = DataLoader()
        X, y = loader.load()

        # ── Étape 2 : Entraînement ────────────────────────────
        trainer = ModelTrainer()
        trainer.train(X, y)

        # ── Étape 3 : Sauvegarde ──────────────────────────────
        trainer.save()

        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║   Entraînement terminé avec succès !         ║")
        logger.info("║   Lancez maintenant : python run_api.py      ║")
        logger.info("╚══════════════════════════════════════════════╝")

    except FileNotFoundError as e:
        logger.error(f"Fichier introuvable : {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Données invalides : {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        raise


if __name__ == "__main__":
    main()
