# backend/main.py - SIMPLIFIÉ
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Essayer d'importer votre modèle
try:
    from xgboost_api import xgboost_model
    print("✅ Modèle XGBoost importé")
    MODEL_READY = True
except ImportError as e:
    print(f"⚠️ Erreur import: {e}")
    xgboost_model = None
    MODEL_READY = False

app = FastAPI(title="SADOP XGBoost")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
def analyze_sql(sql: str):
    """Analyse une requête SQL avec XGBoost"""
    if not MODEL_READY or not xgboost_model:
        return {"error": "Modèle non disponible"}
    
    result = xgboost_model.predict(sql)
    
    if result:
        return {
            "success": True,
            "sql": sql,
            "is_slow": result['is_slow'],
            "confidence": result['confidence'],
            "features": result['features']
        }
    else:
        return {"error": "Erreur d'analyse"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "xgboost": "ready" if MODEL_READY else "not_loaded"
    }

if __name__ == "__main__":
    print("🚀 SADOP avec XGBoost")
    print(f"🤖 Modèle: {'✅ Prêt' if MODEL_READY else '❌ Non chargé'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)