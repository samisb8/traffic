import json
import shutil
from pathlib import Path
import requests

def load_metrics(model_path):
    """Charge les métriques d'un modèle"""
    metrics_path = model_path.parent / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None

def compare_models():
    """Compare nouveau modèle vs modèle en production"""
    
    # Nouveau modèle
    new_model_path = Path("data/model.pkl")
    new_metrics = load_metrics(new_model_path)
    
    # Modèle en production
    prod_model_path = Path("data/model_production.pkl")
    prod_metrics_path = Path("data/model_production_metrics.json")
    
    if not prod_model_path.exists():
        print("🆕 Aucun modèle en production, déploiement du nouveau")
        return True, new_metrics, None
    
    if prod_metrics_path.exists():
        with open(prod_metrics_path) as f:
            prod_metrics = json.load(f)
    else:
        print("⚠️  Métriques production manquantes")
        return True, new_metrics, None
    
    # Comparaison (critère: accuracy)
    new_accuracy = new_metrics.get('accuracy', 0)
    prod_accuracy = prod_metrics.get('accuracy', 0)
    
    print(f"📊 Comparaison modèles:")
    print(f"   Nouveau: {new_accuracy:.4f}")
    print(f"   Production: {prod_accuracy:.4f}")
    
    should_deploy = new_accuracy > prod_accuracy
    return should_deploy, new_metrics, prod_metrics

def deploy_model():
    """Déploie le nouveau modèle si meilleur"""
    print("🚀 Début déploiement modèle...")
    
    # Vérification nouveau modèle
    new_model_path = Path("data/model.pkl")
    if not new_model_path.exists():
        print("❌ Aucun nouveau modèle trouvé")
        return False
    
    # Comparaison
    should_deploy, new_metrics, prod_metrics = compare_models()
    
    if not should_deploy:
        print("❌ Nouveau modèle moins performant, déploiement annulé")
        return False
    
    # Backup modèle actuel
    prod_model_path = Path("data/model_production.pkl")
    if prod_model_path.exists():
        backup_path = Path("data/model_backup.pkl")
        shutil.copy2(prod_model_path, backup_path)
        print(f"💾 Backup créé: {backup_path}")
    
    # Déploiement
    try:
        # Copie nouveau modèle → production
        shutil.copy2(new_model_path, prod_model_path)
        
        # Copie métriques
        shutil.copy2("data/model_metrics.json", "data/model_production_metrics.json")
        
        print("✅ Modèle déployé en production!")
        
        # Notification API (optionnel)
        try:
            response = requests.post("http://localhost:8000/model-updated")
            print("🔄 API notifiée du nouveau modèle")
        except:
            print("⚠️  Impossible de notifier l'API")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur déploiement: {e}")
        return False

if __name__ == "__main__":
    success = deploy_model()
    if success:
        print("🎉 Déploiement réussi!")
    else:
        print("💥 Déploiement échoué!")