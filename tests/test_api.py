"""
tests/test_api.py
──────────────────
Tests d'intégration des endpoints FastAPI.
Utilise le TestClient httpx (sans vrai serveur HTTP).
"""

import pytest


class TestEndpointRoot:
    """Tests du endpoint GET /"""

    def test_root_retourne_200(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200

    def test_root_contient_nom_app(self, api_client):
        data = api_client.get("/").json()
        assert "application" in data


class TestEndpointHealth:
    """Tests du endpoint GET /health"""

    def test_health_retourne_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_modele_charge(self, api_client):
        data = api_client.get("/health").json()
        assert data["modele_charge"] is True
        assert data["status"] == "ok"

    def test_health_contient_classes(self, api_client):
        data = api_client.get("/health").json()
        assert "classes_disponibles" in data
        assert len(data["classes_disponibles"]) == 4


class TestEndpointClasses:
    """Tests du endpoint GET /classes"""

    def test_classes_retourne_200(self, api_client):
        response = api_client.get("/classes")
        assert response.status_code == 200

    def test_classes_contient_4_classes(self, api_client):
        data = api_client.get("/classes").json()
        assert len(data["classes"]) == 4
        assert "Urgence" in data["classes"]
        assert "Commande" in data["classes"]


class TestEndpointPredict:
    """Tests du endpoint POST /predict"""

    def test_predict_retourne_200(self, api_client):
        response = api_client.post("/predict", json={"message": "Test de message"})
        assert response.status_code == 200

    def test_predict_structure_reponse(self, api_client):
        """La réponse doit contenir tous les champs attendus."""
        data = api_client.post(
            "/predict",
            json={"message": "Je veux commander 5 cartons d'eau."}
        ).json()
        champs_requis = {"message", "classe", "confiance", "action_automatique",
                         "reponse_client", "probabilites"}
        assert champs_requis.issubset(data.keys())

    def test_predict_confiance_valide(self, api_client):
        """La confiance doit être entre 0 et 100."""
        data = api_client.post(
            "/predict", json={"message": "Urgent ! Système en panne !"}
        ).json()
        assert 0 <= data["confiance"] <= 100

    def test_predict_message_vide_retourne_422(self, api_client):
        """Un message vide doit retourner une erreur 422."""
        response = api_client.post("/predict", json={"message": ""})
        assert response.status_code == 422

    def test_predict_sans_body_retourne_422(self, api_client):
        """Une requête sans body doit retourner 422."""
        response = api_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_classe_dans_valides(self, api_client):
        """La classe prédite doit être l'une des 4 valides."""
        classes_valides = {"Information", "Réclamation", "Commande", "Urgence"}
        data = api_client.post(
            "/predict", json={"message": "Bonjour, avez-vous du lait ?"}
        ).json()
        assert data["classe"] in classes_valides

    def test_predict_probabilites_4_classes(self, api_client):
        """Les probabilités doivent couvrir les 4 classes."""
        data = api_client.post(
            "/predict", json={"message": "Commande de riz"}
        ).json()
        assert len(data["probabilites"]) == 4

    def test_predict_reponse_client_non_vide(self, api_client):
        """La réponse automatique ne doit pas être vide."""
        data = api_client.post(
            "/predict", json={"message": "J'ai reçu un produit périmé."}
        ).json()
        assert len(data["reponse_client"]) > 10
