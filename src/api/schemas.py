"""
src/api/schemas.py
───────────────────
Modèles Pydantic de validation pour l'API.

Ces schémas définissent :
  - La forme des données d'entrée (requêtes)
  - La forme des données de sortie (réponses)
  - La documentation automatique Swagger
"""

from pydantic import BaseModel, Field, field_validator


# ── Actions automatiques déclenchées par classe ─────────────────
ACTIONS = {
    "Information": "Réponse automatique envoyée au client avec les informations générales.",
    "Commande": "Commande enregistrée dans le système et notification envoyée à l'équipe stock.",
    "Réclamation": "Ticket de réclamation créé et transmis au responsable service client.",
    "Urgence": "ALERTE PRIORITAIRE : Notification immédiate envoyée au responsable. Traitement urgent requis.",
}

# ── Réponses automatiques envoyées au client ─────────────────────
REPONSES_AUTO = {
    "Information": (
        "Bonjour ! Merci pour votre message. "
        "Notre équipe vous répondra avec les informations demandées "
        "dans les plus brefs délais. Horaires : Lun–Sam 8h–20h."
    ),
    "Commande": (
        "Bonjour ! Votre commande a bien été enregistrée. "
        "Vous recevrez une confirmation avec les détails de livraison prochainement. "
        "Merci de votre confiance !"
    ),
    "Réclamation": (
        "Bonjour ! Nous sommes désolés pour cet inconvénient. "
        "Votre réclamation a été transmise à notre responsable service client "
        "qui vous contactera dans les 24h."
    ),
    "Urgence": (
        "Votre demande urgente a été transmise immédiatement au responsable. "
        "Vous serez contacté dans les plus brefs délais. "
        "Merci de votre patience."
    ),
}


# ────────────────────────────────────────────────────────────────
# Schémas de requête
# ────────────────────────────────────────────────────────────────

class DemandeRequest(BaseModel):
    """Corps de la requête POST /predict"""

    message: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Texte du message client à classifier",
        examples=["Urgent ! Le paiement par Wave ne fonctionne pas."],
    )

    @field_validator("message")
    @classmethod
    def message_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le message ne peut pas être vide ou composé uniquement d'espaces.")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Urgent ! Le paiement par Wave ne fonctionne pas."
            }
        }
    }


# ────────────────────────────────────────────────────────────────
# Schémas de réponse
# ────────────────────────────────────────────────────────────────

class DemandeResponse(BaseModel):
    """Réponse du endpoint POST /predict"""

    message: str = Field(..., description="Message original du client")
    classe: str = Field(..., description="Classe prédite par le modèle ML")
    confiance: float = Field(..., description="Probabilité de la classe prédite (en %)")
    action_automatique: str = Field(..., description="Action déclenchée automatiquement")
    reponse_client: str = Field(..., description="Réponse automatique à envoyer au client")
    probabilites: dict[str, float] = Field(
        ..., description="Probabilités pour chaque classe possible (en %)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Urgent ! Le paiement par Wave ne fonctionne pas.",
                "classe": "Urgence",
                "confiance": 94.2,
                "action_automatique": "ALERTE PRIORITAIRE : Notification immédiate envoyée au responsable.",
                "reponse_client": "Votre demande urgente a été transmise immédiatement au responsable.",
                "probabilites": {
                    "Commande": 1.1,
                    "Information": 2.3,
                    "Réclamation": 2.4,
                    "Urgence": 94.2,
                },
            }
        }
    }


class HealthResponse(BaseModel):
    """Réponse du endpoint GET /health"""

    status: str
    modele_charge: bool
    classes_disponibles: list[str]
    version: str
