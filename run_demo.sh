"""
#!/bin/bash

echo "🚀 MLOps Traffic Flow Demo Setup"
echo "================================="

# 1. Installation dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

# 2. Création structure données
echo "📊 Préparation des données..."
mkdir -p data
python mlops/train.py

# 3. Lancement services
echo "🐳 Lancement des services Docker..."
docker-compose up -d

# 4. Attente démarrage
echo "⏳ Attente démarrage services..."
sleep 15

# 5. Tests santé
echo "🏥 Tests de santé..."
curl -f http://localhost:8000/health || echo "❌ API non accessible"
curl -f http://localhost:8501 || echo "❌ Streamlit non accessible"

# 6. Démo MLOps
echo ""
echo "🤖 Démonstration Pipeline MLOps"
echo "================================"

echo "1️⃣ Entraînement nouveau modèle..."
python mlops/train.py

echo "2️⃣ Évaluation et déploiement..."
python mlops/deploy.py

echo "3️⃣ Monitoring..."
python mlops/monitor.py

echo ""
echo "✅ Demo ready!"
echo "🌐 Streamlit: http://localhost:8501"
echo "🚀 API: http://localhost:8000"
echo "📊 MLflow: http://localhost:5000"
echo ""
echo "🎬 Scénario démo:"
echo "  1. Ouvrir Streamlit Dashboard"
echo "  2. Visualiser prédictions temps réel"
echo "  3. Aller sur ML Monitoring"
echo "  4. Lancer re-entraînement"
echo "  5. Observer les métriques"
"""