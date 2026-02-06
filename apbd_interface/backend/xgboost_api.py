# backend/xgboost_api.py - FICHIER SIMPLE POUR L'API
import sys
import os

# Importer depuis votre fichier principal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_model import QueryPerformanceClassifier, SQLFeatureExtractor

class XGBoostAPIModel:
    """Wrapper simple pour utiliser votre modèle dans l'API"""
    def __init__(self):
        self.classifier = None
        self.feature_extractor = SQLFeatureExtractor()
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Charge ou crée un modèle"""
        try:
            # Essayer de charger depuis fichier
            self.classifier = QueryPerformanceClassifier('query_performance_classifier.pkl')
            print("✅ Modèle XGBoost chargé")
        except:
            print("⚠️ Création modèle de test")
            self.classifier = QueryPerformanceClassifier()
            # Pas d'entraînement pour le test
            self.classifier.model = "test_mode"
    
    def predict(self, sql_query):
        """Prédit si une requête est lente"""
        if not self.classifier or self.classifier.model == "test_mode":
            # Mode test - simulation
            features = self.feature_extractor.extract_features(sql_query)
            is_slow = features.get('rows_examined', 0) > 3000
            return {
                'is_slow': is_slow,
                'confidence': 0.85,
                'slow_probability': 0.7 if is_slow else 0.3,
                'fast_probability': 0.3 if is_slow else 0.7,
                'features': features
            }
        
        # Mode réel - utiliser votre vrai modèle
        try:
            features = self.feature_extractor.extract_features(sql_query)
            result = self.classifier.predict_query(features)
            return {
                'is_slow': result['status'] == 'slow',
                'confidence': result['confidence'],
                'slow_probability': result['slow_probability'],
                'fast_probability': result['fast_probability'],
                'features': features
            }
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return None

# Instance globale pour l'API
xgboost_model = XGBoostAPIModel()