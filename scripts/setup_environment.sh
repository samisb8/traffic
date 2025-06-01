echo "🔧 Setup Environnement MLOps Traffic Flow"
echo "=========================================="

# Vérification Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Installez Python 3.9+"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Vérification Docker
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker non trouvé. Installation recommandée pour la demo complète"
else
    echo "🐳 Docker version: $(docker --version)"
fi

# Création environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création environnement virtuel..."
    python3 -m venv venv
fi

echo "🔌 Activation environnement..."
source venv/bin/activate

# Installation dépendances
echo "📥 Installation dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Création structure dossiers
echo "📁 Création structure..."
mkdir -p data logs mlruns

# Génération données et modèle initial
echo "🤖 Génération modèle initial..."
python mlops/train.py

# Variables d'environnement
if [ ! -f ".env" ]; then
    echo "🔧 Création fichier .env..."
    cat > .env << 'EOF'
# MLOps Traffic Flow Configuration
ENV=development
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
MODEL_PATH=data/model.pkl
DATA_PATH=data/traffic_data.csv
MLFLOW_TRACKING_URI=http://localhost:5000
EOF
fi

echo ""
echo "✅ Setup terminé!"
echo ""
echo "🚀 Prochaines étapes:"
echo "  1. source venv/bin/activate  # Activer environnement"
echo "  2. make demo                 # Lancer demo complète"
echo "  3. make docker-up           # Ou juste les services"