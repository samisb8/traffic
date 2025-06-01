echo "🧪 Tests MLOps Traffic Flow"
echo "==========================="

# Tests unitaires
echo "1️⃣ Tests unitaires..."
python -c "
import sys
sys.path.append('.')

# Test imports
try:
    import backend.main
    import streamlit_app.app
    from backend.predictor import TrafficPredictor
    from backend.monitor import ModelMonitor
    print('✅ Imports OK')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)

# Test predictor
try:
    predictor = TrafficPredictor()
    assert predictor.is_loaded(), 'Model not loaded'
    print('✅ Predictor OK')
except Exception as e:
    print(f'❌ Predictor error: {e}')
    exit(1)

# Test monitor
try:
    monitor = ModelMonitor()
    metrics = monitor.get_current_metrics()
    assert 'model_metrics' in metrics, 'Missing model metrics'
    print('✅ Monitor OK')
except Exception as e:
    print(f'❌ Monitor error: {e}')
    exit(1)
"

# Tests MLOps
echo "2️⃣ Tests MLOps..."
python mlops/train.py --test-mode
if [ $? -eq 0 ]; then
    echo "✅ Training OK"
else
    echo "❌ Training failed"
    exit 1
fi

# Tests API (si running)
echo "3️⃣ Tests API..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Health OK"
    
    # Test endpoints
    curl -f http://localhost:8000/predict > /dev/null 2>&1 && echo "✅ Predict endpoint OK" || echo "⚠️  Predict endpoint issue"
    curl -f http://localhost:8000/metrics > /dev/null 2>&1 && echo "✅ Metrics endpoint OK" || echo "⚠️  Metrics endpoint issue"
else
    echo "⚠️  API non accessible (normal si pas démarrée)"
fi

echo ""
echo "✅ Tests terminés!"