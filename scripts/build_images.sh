echo "🐳 Build Images Docker"
echo "====================="

# Création Dockerfile pour API
echo "📝 Création Dockerfile API..."
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY mlops/ ./mlops/
COPY data/ ./data/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "backend/main.py"]
EOF

# Création Dockerfile pour Streamlit
echo "📝 Création Dockerfile Streamlit..."
cat > Dockerfile.streamlit << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app/ ./streamlit_app/
COPY data/ ./data/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Build images
echo "🔨 Build de l'image API..."
docker build -t traffic-api . --no-cache

echo "🔨 Build de l'image Streamlit..."
docker build -f Dockerfile.streamlit -t traffic-streamlit . --no-cache

# Vérification
echo "🔍 Vérification des images..."
docker images | grep traffic

echo "✅ Images construites avec succès!"