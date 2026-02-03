# agent_api.py - FastAPI server for your agent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from deployment import IndexOptimizationAgent
from mysql_utils import get_connection
import mysql.connector

app = FastAPI(title="RL Agent API", description="API for MySQL Index Optimization Agent")

# CORS middleware for interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = None

class OptimizationRequest(BaseModel):
    steps: int = 5
    strategy: str = "balanced"

class ChatRequest(BaseModel):
    message: str
    user_id: str = "interface_user"

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    print("🚀 Starting RL Agent API...")
    
    # Test database connection
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        conn.close()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    # Load agent
    try:
        agent = IndexOptimizationAgent()
        print("✅ RL Agent loaded successfully")
    except Exception as e:
        print(f"⚠  Agent loading warning: {e}")
        agent = None

@app.get("/")
async def root():
    return {
        "service": "RL Agent API",
        "version": "1.0",
        "endpoints": {
            "/health": "Health check",
            "/api/status": "Get database status",
            "/api/optimize": "Run optimization",
            "/api/chat": "Chat with agent"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "agent": "loaded" if agent else "not_loaded"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/status")
async def get_status():
    """Get current database status"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get query performance
        from mysql_utils import measure_query_performance
        perf = measure_query_performance(cursor)
        
        # Get index count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.statistics 
            WHERE table_schema = DATABASE() 
            AND table_name = 'orders'
            AND index_name LIKE 'idx_orders_%'
        """)
        index_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "data": {
                "performance": round(perf, 4),
                "index_count": index_count,
                "max_indexes": 5,
                "database": "pos"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
async def optimize(request: OptimizationRequest):
    """Run optimization with RL agent"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    try:
        # Run optimization
        indexes, query_time = agent.optimize(steps=request.steps)
        recommendations = agent.get_recommendations()
        
        return {
            "status": "success",
            "data": {
                "steps": request.steps,
                "index_count": indexes,
                "query_time": round(query_time, 4),
                "recommendations": recommendations,
                "message": f"Optimization complete. Current indexes: {indexes}/5"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat interface with agent"""
    # Simple rule-based responses for now
    # Your friend can enhance this with LangChain
    
    message_lower = request.message.lower()
    
    if any(word in message_lower for word in ['slow', 'performance', 'lent', 'rapide']):
        response = "I can help optimize your database performance. Would you like me to run an optimization? Use the /api/optimize endpoint or click 'Optimize Now' in the interface."
    
    elif any(word in message_lower for word in ['index', 'indexes', 'indexation']):
        response = "I manage database indexes using Reinforcement Learning. I can create or drop indexes based on query patterns. Current max indexes: 5."
    
    elif any(word in message_lower for word in ['hello', 'hi', 'bonjour', 'help']):
        response = "Hello! I'm your RL-powered database optimization assistant. I can help optimize MySQL indexes, diagnose performance issues, and provide recommendations."
    
    else:
        response = "I'm focused on database optimization. You can ask me about performance issues, index management, or request optimizations."
    
    return {
        "status": "success",
        "data": {
            "message": request.message,
            "response": response,
            "suggested_actions": ["optimize", "diagnose", "status"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "agent_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )