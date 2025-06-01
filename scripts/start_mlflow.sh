#!/bin/bash

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Démarrage MLflow Server pour Casablanca Traffic${NC}"

# Configuration
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
BACKEND_STORE_URI=${BACKEND_STORE_URI:-sqlite:///mlflow.db}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-./mlflow-artifacts}

# Vérification que MLflow est installé
if ! command -v mlflow &> /dev/null; then
    echo -e "${RED}❌ MLflow n'est pas installé${NC}"
    echo -e "${YELLOW}💡 Installation: pip install mlflow${NC}"
    exit 1
fi

# Création du dossier artifacts
echo -e "${BLUE}📁 Création dossier artifacts...${NC}"
mkdir -p ${ARTIFACT_ROOT}

# Vérification du port
if lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port ${MLFLOW_PORT} déjà utilisé${NC}"
    read -p "Voulez-vous arrêter le processus existant? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🔄 Arrêt du processus existant...${NC}"
        kill $(lsof -ti:${MLFLOW_PORT}) 2>/dev/null || true
        sleep 2
    else
        echo -e "${RED}❌ Impossible de démarrer sur le port ${MLFLOW_PORT}${NC}"
        exit 1
    fi
fi

# Affichage de la configuration
echo -e "${GREEN}⚙️  Configuration MLflow:${NC}"
echo -e "   📍 Host: ${MLFLOW_HOST}"
echo -e "   🔌 Port: ${MLFLOW_PORT}"
echo -e "   💾 Backend: ${BACKEND_STORE_URI}"
echo -e "   📦 Artifacts: ${ARTIFACT_ROOT}"

# Démarrage MLflow
echo -e "${GREEN}🚀 Démarrage du serveur MLflow...${NC}"

mlflow server \
    --host ${MLFLOW_HOST} \
    --port ${MLFLOW_PORT} \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --serve-artifacts &

MLFLOW_PID=$!

# Attendre que le serveur démarre
echo -e "${BLUE}⏳ Attente du démarrage du serveur...${NC}"
sleep 5

# Vérification
if kill -0 $MLFLOW_PID 2>/dev/null; then
    echo -e "${GREEN}✅ MLflow démarré avec succès!${NC}"
    echo -e "${GREEN}🌐 Interface: http://localhost:${MLFLOW_PORT}${NC}"
    echo -e "${YELLOW}💡 Pour arrêter: kill ${MLFLOW_PID}${NC}"
    echo -e "${BLUE}📋 PID sauvé dans mlflow.pid${NC}"
    echo $MLFLOW_PID > mlflow.pid
else
    echo -e "${RED}❌ Échec du démarrage MLflow${NC}"
    exit 1
fi

# Garder le script actif
echo -e "${BLUE}🎯 MLflow en cours d'exécution... Ctrl+C pour arrêter${NC}"
wait $MLFLOW_PID