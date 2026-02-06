# backend/main.py - CORRIGÉ
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

# Ajouter le dossier courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ... au début du fichier (après les imports existants) ...

# ==================== REMPLACER CE BLOC ====================
# Ajouter le dossier courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Maintenant importer depuis agent
try:
    from agent.langchain_agent import sadop_agent
    print("✅ Agent LangChain chargé")
except ImportError as e:
    print(f"⚠️ Agent LangChain non disponible: {e}")
    sadop_agent = None

# ==================== PAR CE CODE ====================

# Ajouter le chemin du dossier agent indépendant
current_dir = os.path.dirname(os.path.abspath(__file__))  # backend/
project_root = os.path.dirname(os.path.dirname(current_dir))  # apbd_interface/
agent_dir = os.path.join(project_root, "agent")  # apbd_interface/agent

if os.path.exists(agent_dir):
    sys.path.insert(0, agent_dir)  # Ajouter en première position
    print(f"✅ Dossier agent trouvé: {agent_dir}")
else:
    print(f"❌ Dossier agent introuvable: {agent_dir}")
    agent_dir = None

# Essayer d'importer l'agent LangChain
try:
    if agent_dir:
        # Option 1: Importer depuis le dossier agent
        from langchain_agent import sadop_agent
        print("✅ Agent LangChain chargé depuis dossier indépendant")
    else:
        raise ImportError("Dossier agent non trouvé")
        
except ImportError as e:
    print(f"⚠️ Agent LangChain non disponible: {e}")
    
    # ==================== CODE LANGCHAIN DIRECTEMENT DANS MAIN.PY ====================
    # Créer un agent simple directement
    import asyncio
    
    class SimpleSADOPAgent:
        """Agent simple intégré directement dans main.py"""
        
        def __init__(self):
            print("✅ Agent simple initialisé (intégré dans main.py)")
        
        async def query(self, user_input: str):
            """Analyse une requête SQL"""
            sql_upper = user_input.upper()
            
            # Détecter si c'est une requête SQL
            sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM ", "WHERE "]
            if not any(k in sql_upper for k in sql_keywords):
                return "❓ Veuillez fournir une requête SQL valide à analyser."
            
            # Analyse simple
            issues = []
            suggestions = []
            
            if "SELECT *" in sql_upper:
                issues.append("Utilise `SELECT *` (récupère toutes les colonnes)")
                suggestions.append("Spécifiez uniquement les colonnes nécessaires")
            
            if "WHERE" not in sql_upper and "LIMIT" not in sql_upper:
                issues.append("Pas de clause WHERE ou LIMIT")
                suggestions.append("Ajoutez une clause WHERE pour filtrer les résultats")
            
            if "JOIN" in sql_upper:
                issues.append("Contient des jointures")
                suggestions.append("Vérifiez que les colonnes de jointure sont indexées")
            
            # Construire la réponse
            response = [
                "## 🔍 Analyse SQL SADOP",
                "",
                "### 📝 Requête analysée",
                "```sql",
                user_input,
                "```",
                ""
            ]
            
            if issues:
                response.append("### ⚠️ Problèmes détectés")
                for issue in issues:
                    response.append(f"- {issue}")
                response.append("")
            
            response.append("### 💡 Recommandations")
            for suggestion in suggestions:
                response.append(f"- {suggestion}")
            
            # Ajouter des conseils généraux
            response.append("")
            response.append("### 🔧 Conseils d'optimisation")
            response.append("1. **Utilisez EXPLAIN** pour voir le plan d'exécution :")
            response.append(f"   ```sql")
            response.append(f"   EXPLAIN {user_input}")
            response.append(f"   ```")
            response.append("2. **Ajoutez des index** sur les colonnes filtrées")
            response.append("3. **Évitez SELECT *** - spécifiez les colonnes")
            
            return "\n".join(response)
    
    # Créer une instance de l'agent intégré
    sadop_agent = SimpleSADOPAgent()
    print("✅ Agent simple créé (intégré dans main.py)")
    
    # ==================== FIN DU CODE LANGCHAIN ====================

# ==================== CONTINUER AVEC LE RESTE DU FICHIER ====================
app = FastAPI(title="SADOP API", version="2.0")


# Autoriser Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines pour le test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"

@app.post("/chat")
async def chat_with_agent(request: QueryRequest):
    """Endpoint principal pour interagir avec l'agent SADOP"""
    try:
        print(f"📩 Question reçue: {request.question[:100]}...")
        
        if sadop_agent is None:
            return {
                "success": True,
                "question": request.question,
                "response": f"✅ Réponse de test SADOP\n\n**Question:** {request.question}\n\n**Statut:** L'agent LangChain n'est pas encore configuré. Le backend fonctionne correctement.",
                "source": "SADOP Test"
            }
        
        # Utiliser l'agent LangChain
        response = await sadop_agent.query(request.question)
        
        return {
            "success": True,
            "question": request.question,
            "response": response,
            "source": "SADOP Agent"
        }
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {
            "success": False,
            "error": str(e),
            "response": f"Erreur lors du traitement: {str(e)}"
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "SADOP API",
        "version": "2.0",
        "backend": "running"
    }

@app.get("/")
async def root():
    return {
        "message": "🚀 SADOP API v2.0 - Système Autonome de Diagnostic et d'Optimisation",
        "endpoints": {
            "GET /": "Cette page",
            "GET /health": "Vérifier l'état",
            "POST /chat": "Chat avec l'agent IA",
            "GET /test": "Test simple"
        }
    }

@app.get("/test")
async def test():
    return {"message": "Backend SADOP fonctionne !"}

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 SADOP BACKEND v2.0")
    print("=" * 50)
    print("🌐 URLs:")
    print("  • API: http://localhost:8000")
    print("  • Health: http://localhost:8000/health")
    print("  • Test: http://localhost:8000/test")
    print("=" * 50)
    print("💡 Test rapide avec curl:")
    print('  curl -X POST http://localhost:8000/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d "{\\"question\\":\\"Test de connexion\\"}"')
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")