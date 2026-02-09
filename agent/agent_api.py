# flask_api.py - API Flask pour ton agent
from flask import Flask, request, jsonify
from flask_cors import CORS
from deployment import IndexOptimizationAgent
import time
import sys
import os

# Configure pour éviter les warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines (pour l'interface)

print("=" * 60)
print("🤖 AGENT RL API - FLASK VERSION")
print("=" * 60)

# Initialise l'agent
try:
    agent = IndexOptimizationAgent()
    print("✅ Agent RL initialisé")
except Exception as e:
    print(f"⚠  Erreur initialisation agent: {e}")
    agent = None

@app.route('/')
def home():
    return jsonify({
        "service": "Agent RL d'optimisation MySQL",
        "version": "1.0",
        "author": "APBD Team",
        "endpoints": {
            "/health": "État du service",
            "/api/status": "État de la base",
            "/api/optimize": "Lancer l'optimisation",
            "/api/recommendations": "Recommandations",
            "/api/chat": "Chat avec l'agent"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Vérifie que tout fonctionne"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "agent": "loaded" if agent and hasattr(agent, 'model_loaded') and agent.model_loaded else "simulation",
        "message": "🤖 Agent RL prêt à optimiser !"
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """État actuel de la base de données"""
    if not agent:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé"
        }), 500
    
    try:
        status = agent.get_status()
        return jsonify({
            "success": True,
            "data": status,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Exécute l'optimisation RL"""
    if not agent:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé"
        }), 500
    
    try:
        data = request.get_json(silent=True) or {}
        steps = data.get('steps', 5)
        strategy = data.get('strategy', 'balanced')
        
        print(f"🔄 Optimisation demandée: {steps} étapes")
        
        result = agent.optimize(steps=steps)
        
        return jsonify({
            "success": True,
            "data": result,
            "message": "Optimisation terminée avec succès",
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Recommandations d'index"""
    if not agent:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé"
        }), 500
    
    try:
        recommendations = agent.get_recommendations()
        
        return jsonify({
            "success": True,
            "data": recommendations,
            "message": "Recommandations générées",
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Interface de chat avec l'agent"""
    if not agent:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé"
        }), 500
    
    try:
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').lower()
        user_id = data.get('user_id', 'interface_user')
        
        # Réponses basées sur les mots-clés
        if any(word in message for word in ['slow', 'lent', 'performance', 'rapid']):
            response = "Je peux optimiser les performances de votre base MySQL. Voulez-vous que je lance une optimisation automatique des index ?"
            actions = ["optimize"]
            
        elif any(word in message for word in ['index', 'indexes', 'indexation']):
            current_idx = agent.current_indexes if hasattr(agent, 'current_indexes') else 2
            response = f"Je gère les index avec l'apprentissage par renforcement. Je peux créer ou supprimer des index intelligemment. Actuellement, vous avez {current_idx}/5 index."
            actions = ["status", "recommendations"]
            
        elif any(word in message for word in ['hello', 'bonjour', 'hi', 'salut']):
            response = "Bonjour ! Je suis votre assistant RL pour l'optimisation de bases MySQL. Je peux optimiser vos index, diagnostiquer les problèmes et vous donner des recommandations."
            actions = ["help"]
            
        elif any(word in message for word in ['help', 'aide', 'que faire']):
            response = "Je peux :\n1. Optimiser automatiquement vos index\n2. Analyser les performances\n3. Donner des recommandations\n4. Dialoguer sur l'optimisation MySQL\nDites-moi ce que vous voulez faire !"
            actions = ["optimize", "status", "recommendations"]
            
        else:
            response = "Je suis spécialisé dans l'optimisation MySQL avec RL. Posez-moi des questions sur les performances, les index, ou demandez-moi d'optimiser votre base."
            actions = []
        
        return jsonify({
            "success": True,
            "data": {
                "query": data.get('message', ''),
                "response": response,
                "suggested_actions": actions
            },
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

if __name__ == '__main__':
    print("\n📡 Endpoints disponibles:")
    print("  - http://localhost:5000/")
    print("  - http://localhost:5000/health")
    print("  - http://localhost:5000/api/status")
    print("  - http://localhost:5000/api/optimize")
    print("  - http://localhost:5000/api/recommendations")
    print("  - http://localhost:5000/api/chat")
    
    print("\n🤝 Pour l'interface de ton amie:")
    print("  Elle pourra appeler ces endpoints depuis son interface Streamlit")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)