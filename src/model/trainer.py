"""
src/model/trainer.py
─────────────────────
Construction du pipeline ML, entraînement et évaluation.

Pipeline :
    TF-IDF (unigrammes + bigrammes) → Régression Logistique

Responsabilités :
  - Construire le pipeline scikit-learn
  - Découper les données train/test
  - Entraîner le modèle
  - Évaluer et afficher les métriques
  - Sauvegarder le modèle entraîné avec joblib
"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from config.settings import settings
from src.utils.logger import logger


class ModelTrainer:
    """
    Orchestre l'entraînement complet du modèle de classification.

    Exemple d'utilisation :
        trainer = ModelTrainer()
        trainer.train(X, y)
        trainer.save()
    """

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.X_test = None
        self.y_test = None

    # ──────────────────────────────────────────────────────────────
    # Méthode principale
    # ──────────────────────────────────────────────────────────────

    def train(self, X: pd.Series, y: pd.Series) -> "ModelTrainer":
        """
        Entraîne le pipeline ML complet.

        Args:
            X : pd.Series — textes des demandes
            y : pd.Series — classes cibles

        Returns:
            self (chaînable)
        """
        logger.info("Démarrage de l'entraînement du modèle...")

        # 1. Découpage des données
        X_train, X_test, y_train, y_test = self._split(X, y)
        self.X_test = X_test
        self.y_test = y_test

        # 2. Construction du pipeline
        self.pipeline = self._build_pipeline()
        logger.info("Pipeline construit : TF-IDF → Régression Logistique")

        # 3. Entraînement
        self.pipeline.fit(X_train, y_train)
        logger.success("Modèle entraîné avec succès !")

        # 4. Évaluation
        self._evaluate(X_test, y_test)
        self._cross_validate(X, y)

        return self

    def save(self) -> None:
        """Sauvegarde le modèle entraîné dans le chemin configuré."""
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné. Appelez .train() d'abord.")

        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, settings.model_path)
        logger.success(f"Modèle sauvegardé : {settings.model_path}")

    # ──────────────────────────────────────────────────────────────
    # Méthodes privées
    # ──────────────────────────────────────────────────────────────

    def _split(self, X: pd.Series, y: pd.Series):
        """Découpe le dataset en ensembles d'entraînement et de test."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y,        # Garantit la même proportion de classes
        )
        logger.info(
            f"Données découpées — Train : {len(X_train)} | Test : {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test

    def _build_pipeline(self) -> Pipeline:
        """
        Construit le pipeline scikit-learn.

        Étape 1 — TF-IDF :
            Transforme chaque texte en vecteur numérique.
            Les bigrammes capturent des expressions comme "paiement wave",
            "produit périmé", "commande urgente".

        Étape 2 — Régression Logistique :
            Classifieur linéaire rapide, très efficace sur du texte court.
            Renvoie des probabilités par classe (utile pour la confiance).
        """
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(
                    settings.tfidf_ngram_min,
                    settings.tfidf_ngram_max,
                ),
                max_features=settings.tfidf_max_features,
                sublinear_tf=True,      # log(1 + tf) : réduit l'impact des mots très fréquents
                strip_accents="unicode",
                analyzer="word",
                min_df=1,               # accepte les mots rares (dataset petit)
            )),
            ("clf", LogisticRegression(
                C=settings.lr_c,
                max_iter=settings.lr_max_iter,
                solver="lbfgs",
                class_weight="balanced", # Compense les éventuels déséquilibres de classes
            )),
        ])

    def _evaluate(self, X_test: pd.Series, y_test: pd.Series) -> None:
        """Calcule et affiche les métriques sur le jeu de test."""
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info("── Résultats sur le jeu de test ─────────────────")
        logger.info(f"   Accuracy globale : {accuracy * 100:.1f}%")
        logger.info("── Rapport par classe ───────────────────────────")

        report = classification_report(y_test, y_pred)
        for line in report.split("\n"):
            if line.strip():
                logger.info(f"   {line}")

        logger.info("── Matrice de confusion ─────────────────────────")
        classes = self.pipeline.classes_
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        # En-tête
        header = "         " + "  ".join(f"{c[:4]:>6}" for c in classes)
        logger.info(header)
        for i, row in enumerate(cm):
            logger.info(f"   {classes[i]:<12}" + "  ".join(f"{v:>6}" for v in row))
        logger.info("─────────────────────────────────────────────────")

    def _cross_validate(self, X: pd.Series, y: pd.Series) -> None:
        """Validation croisée 5 folds pour estimer la performance réelle."""
        logger.info("Validation croisée 5 folds en cours...")
        scores = cross_val_score(
            self.pipeline, X, y,
            cv=5,
            scoring="f1_weighted",
        )
        logger.info(
            f"F1-score moyen (CV-5) : {scores.mean() * 100:.1f}% "
            f"± {scores.std() * 100:.1f}%"
        )
