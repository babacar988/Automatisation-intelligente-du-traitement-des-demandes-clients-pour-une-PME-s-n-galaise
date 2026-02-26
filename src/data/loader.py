"""
src/data/loader.py
───────────────────
Chargement, validation et statistiques du dataset.

Responsabilités :
  - Charger le CSV depuis le chemin configuré
  - Valider la structure (colonnes, valeurs nulles, classes)
  - Retourner X (textes) et y (labels) prêts pour l'entraînement
  - Afficher des statistiques sur la distribution des classes
"""

import pandas as pd
from pathlib import Path
from config.settings import settings
from src.utils.logger import logger

# Classes autorisées dans le dataset
CLASSES_VALIDES = {"Information", "Réclamation", "Commande", "Urgence"}


class DataLoader:
    """
    Charge et valide le dataset de demandes clients.

    Exemple d'utilisation :
        loader = DataLoader()
        X, y = loader.load()
    """

    def __init__(self, path: Path = None):
        self.path = path or settings.dataset_path

    # ──────────────────────────────────────────────────────────────
    # Méthode principale
    # ──────────────────────────────────────────────────────────────

    def load(self) -> tuple[pd.Series, pd.Series]:
        """
        Charge le CSV, valide les données et retourne (X, y).

        Returns:
            X : pd.Series — textes des demandes clients
            y : pd.Series — classes cibles (Information, Réclamation, etc.)

        Raises:
            FileNotFoundError : si le fichier CSV est introuvable
            ValueError        : si les données sont invalides
        """
        logger.info(f"Chargement du dataset : {self.path}")

        df = self._read_csv()
        self._validate(df)
        self._print_stats(df)

        X = df["texte"].str.strip()
        y = df["classe"].str.strip()

        logger.success(f"{len(df)} exemples chargés avec succès.")
        return X, y

    # ──────────────────────────────────────────────────────────────
    # Méthodes privées
    # ──────────────────────────────────────────────────────────────

    def _read_csv(self) -> pd.DataFrame:
        """Lit le fichier CSV et lève une erreur claire si absent."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.path}\n"
                "Vérifiez le chemin dans votre fichier .env (DATASET_PATH)."
            )
        return pd.read_csv(self.path)

    def _validate(self, df: pd.DataFrame) -> None:
        """Vérifie la structure et la qualité du dataset."""

        # 1. Colonnes requises
        colonnes_requises = {"texte", "classe"}
        colonnes_manquantes = colonnes_requises - set(df.columns)
        if colonnes_manquantes:
            raise ValueError(
                f"Colonnes manquantes dans le CSV : {colonnes_manquantes}\n"
                "Le fichier doit avoir les colonnes : 'texte' et 'classe'."
            )

        # 2. Valeurs nulles
        nulls = df[["texte", "classe"]].isnull().sum()
        if nulls.any():
            raise ValueError(
                f"Valeurs nulles détectées :\n{nulls[nulls > 0]}\n"
                "Nettoyez le dataset avant l'entraînement."
            )

        # 3. Classes inconnues
        classes_inconnues = set(df["classe"].unique()) - CLASSES_VALIDES
        if classes_inconnues:
            raise ValueError(
                f"Classes inconnues dans le dataset : {classes_inconnues}\n"
                f"Classes autorisées : {CLASSES_VALIDES}"
            )

        # 4. Textes trop courts
        textes_courts = df[df["texte"].str.len() < 5]
        if not textes_courts.empty:
            logger.warning(
                f"{len(textes_courts)} texte(s) très courts détectés (< 5 caractères). "
                "Vérifiez leur pertinence."
            )

        logger.info("Validation du dataset : OK")

    def _print_stats(self, df: pd.DataFrame) -> None:
        """Affiche la distribution des classes dans les logs."""
        logger.info("── Distribution des classes ──────────────────────")
        distribution = df["classe"].value_counts()
        for classe, count in distribution.items():
            pct = count / len(df) * 100
            logger.info(f"   {classe:<15} : {count:>3} exemples ({pct:.1f}%)")
        logger.info(f"   {'TOTAL':<15} : {len(df):>3} exemples")
        logger.info("──────────────────────────────────────────────────")
