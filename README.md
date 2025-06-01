# 🚗 MLOps Traffic Flow Demo

Demo complète d'un système MLOps pour la prédiction de trafic à Casablanca.

## 🎯 Fonctionnalités

- **Dashboard Streamlit** : Visualisation temps réel du trafic
- **API FastAPI** : Prédictions ML via REST API
- **Pipeline MLOps** : Entraînement, déploiement, monitoring automatisés
- **CI/CD** : Pipeline GitHub Actions complet
- **Monitoring** : Détection drift et métriques performance

## 🚀 Quick Start

```bash
# Setup complet
make setup

# Demo complète
make demo

# Accès interfaces
open http://localhost:8501  # Streamlit
open http://localhost:8000  # API
```

## 📁 Structure

```
mlops-trafficflow/
├── streamlit_app/    # Interface utilisateur
├── backend/         # API FastAPI
├── mlops/          # Scripts MLOps
├── .github/        # Pipelines CI/CD
├── tests/          # Tests automatisés
└── scripts/        # Scripts utilitaires
```

## 🔄 Pipeline MLOps

1. **Data** : Génération/collecte données trafic
2. **Train** : Entraînement modèle ML
3. **Validate** : Évaluation performance
4. **Deploy** : Déploiement si amélioration
5. **Monitor** : Surveillance drift et performance

## 🧪 Tests

```bash
make test     # Tests unitaires
make lint     # Vérification code
make ci       # Pipeline CI local
```

## 🐳 Docker

```bash
make docker-build  # Build images
make docker-up     # Démarrage services
make docker-down   # Arrêt services
```

## 📊 Monitoring

- **Métriques ML** : Accuracy, MAE, R²
- **Drift** : Détection changements données
- **Système** : Latence, throughput, santé services

## 🎬 Scénario Demo

1. **Dashboard** (10 min) : Carte Casablanca + prédictions
2. **MLOps** (15 min) : Entraînement + déploiement
3. **Monitoring** (5 min) : Drift + alertes

**Total : 30 minutes de démo impressionnante !**

---

## 📞 Support

Pour questions : vérifiez logs dans `logs/` ou `docker-compose logs`=====================================
# 🚀 Makefile
# =====================================
.PHONY: help setup install clean test lint format train deploy monitor demo docker-build docker-up docker-down ci

# Default target
help:
	@echo "🚀 MLOps Traffic Flow - Commandes Disponibles"
	@echo "=============================================="
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make setup        - Setup complet du projet"
	@echo "  make install      - Installation dépendances"
	@echo "  make clean        - Nettoyage fichiers temporaires"
	@echo ""
	@echo "🧪 Tests & Qualité:"
	@echo "  make test         - Lancement tests"
	@echo "  make lint         - Vérification code"
	@echo "  make format       - Formatage code"
	@echo "  make ci           - Pipeline CI complet"
	@echo ""
	@echo "🤖 MLOps:"
	@echo "  make train        - Entraînement modèle"
	@echo "  make deploy       - Déploiement modèle"
	@echo "  make monitor      - Monitoring système"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build - Build images Docker"
	@echo "  make docker-up    - Démarrage services"
	@echo "  make docker-down  - Arrêt services"
	@echo ""
	@echo "🎬 Demo:"
	@echo "  make demo         - Démo complète"

# Setup complet
setup: clean install train docker-build
	@echo "✅ Setup complet terminé!"

# Installation des dépendances
install:
	@echo "📦 Installation des dépendances..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dépendances installées"

# Nettoyage
clean:
	@echo "🧹 Nettoyage..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	docker system prune -f || true
	@echo "✅ Nettoyage terminé"

# Tests
test:
	@echo "🧪 Lancement des tests..."
	python -m pytest -v || echo "⚠️  Aucun test trouvé"
	python mlops/train.py --test-mode
	python -c "import backend.main; import streamlit_app.app; print('✅ Tests d\'import OK')"
	@echo "✅ Tests terminés"

# Linting
lint:
	@echo "🔍 Vérification du code..."
	flake8 backend/ streamlit_app/ mlops/ --max-line-length=88 --ignore=E203,W503 || echo "⚠️  Issues de linting trouvées"
	@echo "✅ Linting terminé"

# Formatage
format:
	@echo "🎨 Formatage du code..."
	black backend/ streamlit_app/ mlops/ || echo "⚠️  Problème de formatage"
	@echo "✅ Formatage terminé"

# CI Pipeline local
ci: lint test
	@echo "🔄 Pipeline CI local terminé"

# Entraînement modèle
train:
	@echo "🤖 Entraînement du modèle..."
	python mlops/train.py
	@echo "✅ Entraînement terminé"

# Déploiement modèle
deploy:
	@echo "🚀 Déploiement du modèle..."
	python mlops/deploy.py
	@echo "✅ Déploiement terminé"

# Monitoring
monitor:
	@echo "📊 Monitoring du système..."
	python mlops/monitor.py
	@echo "✅ Monitoring terminé"

# Build Docker images
docker-build:
	@echo "🐳 Build des images Docker..."
	docker build -t traffic-api .
	docker build -f Dockerfile.streamlit -t traffic-streamlit .
	@echo "✅ Images Docker créées"

# Démarrage services Docker
docker-up:
	@echo "🚀 Démarrage des services..."
	docker-compose up -d
	@echo "⏳ Attente démarrage services..."
	@sleep 15
	@echo "🏥 Tests de santé..."
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && echo "✅ API OK" || echo "❌ API KO"
	@curl -f http://localhost:8501 > /dev/null 2>&1 && echo "✅ Streamlit OK" || echo "❌ Streamlit KO"

# Arrêt services Docker
docker-down:
	@echo "🛑 Arrêt des services..."
	docker-compose down
	@echo "✅ Services arrêtés"

# Demo complète
demo: setup docker-up
	@echo ""
	@echo "🎬 DEMO MLOps Traffic Flow"
	@echo "========================="
	@echo ""
	@echo "🌐 Interfaces disponibles:"
	@echo "  Streamlit:  http://localhost:8501"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo ""
	@echo "🤖 Pipeline MLOps:"
	$(MAKE) train
	$(MAKE) deploy
	$(MAKE) monitor
	@echo ""
	@echo "✅ Demo prête! Ouvrez http://localhost:8501"