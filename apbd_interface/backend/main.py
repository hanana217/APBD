# backend/main.py - CORRIGÉ
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

# Ajouter le dossier courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Maintenant importer depuis agent
try:
    from agent.langchain_agent import sadop_agent
    print("✅ Agent LangChain chargé")
except ImportError as e:
    print(f"⚠️ Agent LangChain non disponible: {e}")
    sadop_agent = None

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