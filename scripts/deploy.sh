echo "🚀 Déploiement MLOps Traffic Flow"
echo "================================="

# Variables
ENVIRONMENT=${1:-production}
BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"

echo "📋 Environnement: $ENVIRONMENT"

# Création backup
if [ "$ENVIRONMENT" = "production" ]; then
    echo "💾 Création backup..."
    mkdir -p $BACKUP_DIR
    
    # Backup modèles
    if [ -f "data/model_production.pkl" ]; then
        cp data/model_production.pkl $BACKUP_DIR/
        cp data/model_production_metrics.json $BACKUP_DIR/ 2>/dev/null || true
        echo "✅ Backup modèle créé"
    fi
    
    # Backup config
    cp docker-compose.yml $BACKUP_DIR/ 2>/dev/null || true
fi

# Arrêt services existants
echo "🛑 Arrêt des services existants..."
docker-compose down 2>/dev/null || true

# Nettoyage images obsolètes
echo "🧹 Nettoyage images obsolètes..."
docker image prune -f

# Build nouvelles images
echo "🔨 Build des nouvelles images..."
./scripts/build_images.sh

# Déploiement modèle
echo "🤖 Déploiement du modèle..."
python mlops/deploy.py

# Configuration spécifique à l'environnement
if [ "$ENVIRONMENT" = "staging" ]; then
    echo "🧪 Configuration staging..."
    export API_PORT=8010
    export STREAMLIT_PORT=8511
    docker-compose -f docker-compose.staging.yml up -d
elif [ "$ENVIRONMENT" = "production" ]; then
    echo "🚀 Configuration production..."
    export API_PORT=8000
    export STREAMLIT_PORT=8501
    docker-compose up -d
else
    echo "🛠️  Configuration développement..."
    docker-compose up -d
fi

# Attente démarrage
echo "⏳ Attente démarrage des services..."
sleep 30

# Tests de santé
echo "🏥 Tests de santé..."
API_URL="http://localhost:${API_PORT:-8000}"
STREAMLIT_URL="http://localhost:${STREAMLIT_PORT:-8501}"

# Test API
for i in {1..5}; do
    if curl -f $API_URL/health > /dev/null 2>&1; then
        echo "✅ API healthy"
        break
    fi
    echo "⏳ Attente API... ($i/5)"
    sleep 10
done

# Test Streamlit
curl -f $STREAMLIT_URL > /dev/null 2>&1 && echo "✅ Streamlit healthy" || echo "⚠️  Streamlit check failed"

# Test endpoints API
if curl -f $API_URL/predict > /dev/null 2>&1; then
    echo "✅ Endpoint /predict OK"
else
    echo "❌ Endpoint /predict failed"
    exit 1
fi

# Monitoring post-déploiement
echo "📊 Monitoring post-déploiement..."
python mlops/monitor.py

# Logs de déploiement
echo "📝 Génération logs de déploiement..."
cat > logs/deployment_$(date +%Y%m%d_%H%M%S).log << EOF
Deployment Log
==============
Date: $(date)
Environment: $ENVIRONMENT
API URL: $API_URL
Streamlit URL: $STREAMLIT_URL
Backup Location: $BACKUP_DIR
Status: SUCCESS
EOF

echo ""
echo "✅ Déploiement réussi!"
echo "🌐 API: $API_URL"
echo "📊 Streamlit: $STREAMLIT_URL"
echo "📋 API Docs: $API_URL/docs"