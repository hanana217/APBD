# backend/main.py - API PRINCIPALE SADOP AVEC RL
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os
import asyncio
import mysql.connector
from datetime import datetime
import pandas as pd
import time
import json
import numpy as np

# ==================== CONFIGURATION ====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration MySQL
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos',
    'autocommit': True
}

# Seuil pour requête lente
SLOW_QUERY_THRESHOLD = 0.5

# Configuration RL
RL_CONFIG = {
    'max_indexes': 5,
    'creation_cost': 0.02,
    'drop_penalty': 0.01,
    'episode_length': 25
}
# ==================== OPTIMISEUR SQL ====================
class SQLOptimizer:
    """Optimiseur de requêtes SQL"""
    
    def __init__(self):
        print("🔧 Optimiseur SQL initialisé")
    
    def optimize_query(self, sql: str) -> Dict[str, Any]:
        """Optimise une requête SQL lente"""
        sql_upper = sql.upper()
        
        # Analyse de la requête
        analysis = self._analyze_query(sql)
        
        # Génération de la version optimisée
        optimized_sql = self._generate_optimized_version(sql, analysis)
        
        # Explications des optimisations
        optimizations = self._get_optimizations_applied(sql, optimized_sql, analysis)
        
        # Estimation d'amélioration
        improvement = self._estimate_improvement(analysis)
        
        return {
            "original_sql": sql,
            "optimized_sql": optimized_sql,
            "optimizations": optimizations,
            "estimated_improvement": improvement,
            "analysis": analysis
        }
    
    def _analyze_query(self, sql: str) -> Dict[str, Any]:
        """Analyse une requête SQL pour identifier les problèmes"""
        sql_upper = sql.upper()
        
        issues = []
        suggestions = []
        
        # Détection de SELECT *
        if "SELECT *" in sql_upper:
            issues.append("SELECT *")
            suggestions.append("Remplacer par les colonnes spécifiques nécessaires")
        
        # Détection de jointures sans index
        if "JOIN" in sql_upper:
            issues.append("Jointure(s) détectée(s)")
            suggestions.append("Vérifier les index sur les colonnes de jointure")
        
        # Détection de WHERE sans index
        if "WHERE" in sql_upper:
            issues.append("Clause WHERE présente")
            suggestions.append("Ajouter des index sur les colonnes filtrées")
        
        # Détection de ORDER BY sans LIMIT
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            issues.append("ORDER BY sans LIMIT")
            suggestions.append("Ajouter LIMIT pour éviter les scans complets")
        
        # Détection de sous-requêtes
        if sql_upper.count("SELECT") > 1:
            issues.append("Sous-requêtes détectées")
            suggestions.append("Considérer l'utilisation de JOIN ou de CTE")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "complexity": self._calculate_complexity(sql_upper)
        }
    
    def _generate_optimized_version(self, sql: str, analysis: Dict) -> str:
        """Génère une version optimisée de la requête"""
        sql_upper = sql.upper()
        optimized = sql
        
        # 1. Remplacer SELECT * si nécessaire
        if "SELECT *" in sql_upper and "FROM" in sql_upper:
            # Essayer d'identifier les colonnes
            table_match = self._extract_table_name(sql)
            if table_match:
                optimized = sql.replace("SELECT *", f"SELECT id, ... /* spécifiez les colonnes */")
        
        # 2. Ajouter LIMIT si ORDER BY sans LIMIT
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            if ";" in optimized:
                optimized = optimized.replace(";", " LIMIT 100;")
            else:
                optimized += " LIMIT 100"
        
        # 3. Simplifier les sous-requêtes
        if sql_upper.count("SELECT") > 1:
            optimized = self._simplify_subqueries(optimized)
        
        return optimized
    
    def _get_optimizations_applied(self, original: str, optimized: str, analysis: Dict) -> List[Dict]:
        """Liste les optimisations appliquées"""
        optimizations = []
        
        if "SELECT *" in original.upper() and "SELECT *" not in optimized.upper():
            optimizations.append({
                "type": "SELECT_STAR",
                "description": "Remplacement de SELECT * par des colonnes spécifiques",
                "impact": "Réduction du transfert de données",
                "sql_example": "-- Remplacez:\n-- SELECT * FROM table\n-- Par:\n-- SELECT id, col1, col2 FROM table"
            })
        
        if "ORDER BY" in original.upper() and "LIMIT" in optimized.upper() and "LIMIT" not in original.upper():
            optimizations.append({
                "type": "ADD_LIMIT",
                "description": "Ajout de LIMIT aux requêtes avec ORDER BY",
                "impact": "Évite les scans complets de table",
                "sql_example": "-- Ajoutez LIMIT:\n-- SELECT ... ORDER BY date\n-- Devient:\n-- SELECT ... ORDER BY date LIMIT 100"
            })
        
        if original.upper().count("SELECT") > 1 and optimized.upper().count("SELECT") < original.upper().count("SELECT"):
            optimizations.append({
                "type": "SIMPLIFY_SUBQUERIES",
                "description": "Simplification des sous-requêtes",
                "impact": "Réduction de la complexité d'exécution",
                "sql_example": "-- Remplacez:\n-- SELECT * FROM (SELECT ...) as sub\n-- Par:\n-- SELECT ... FROM table"
            })
        
        # Recommandations d'index
        table_name = self._extract_table_name(original)
        if table_name:
            optimizations.append({
                "type": "INDEX_RECOMMENDATION",
                "description": f"Création d'index sur la table {table_name}",
                "impact": "Accélération significative des requêtes",
                "sql_example": f"-- Pour les WHERE:\nCREATE INDEX idx_{table_name}_filter ON {table_name}(colonne_filtree);\n-- Pour les JOIN:\nCREATE INDEX idx_{table_name}_join ON {table_name}(colonne_jointure);"
            })
        
        return optimizations
    
    def _estimate_improvement(self, analysis: Dict) -> Dict[str, float]:
        """Estime l'amélioration de performance"""
        complexity = analysis.get("complexity", 1.0)
        issues_count = len(analysis.get("issues", []))
        
        # Estimation basée sur la complexité et les problèmes
        base_improvement = 0.3  # 30% d'amélioration de base
        issue_penalty = issues_count * 0.1  # 10% par problème
        
        estimated = max(0.1, base_improvement - issue_penalty)
        
        return {
            "estimated_percent": round(estimated * 100, 1),
            "confidence": round(1.0 - (issues_count * 0.05), 2),
            "expected_time_reduction": f"{estimated * 100:.1f}%"
        }
    
    def _calculate_complexity(self, sql: str) -> float:
        """Calcule la complexité d'une requête"""
        complexity = 1.0
        
        # Facteurs de complexité
        if "JOIN" in sql:
            complexity += 0.5 * sql.count("JOIN")
        
        if "WHERE" in sql:
            complexity += 0.3 * sql.count("WHERE")
        
        if "GROUP BY" in sql:
            complexity += 0.4
        
        if "ORDER BY" in sql:
            complexity += 0.3
        
        if "SELECT" in sql and sql.count("SELECT") > 1:
            complexity += 0.5 * (sql.count("SELECT") - 1)
        
        return round(complexity, 2)
    
    def _extract_table_name(self, sql: str) -> Optional[str]:
        """Extrait le nom de la table principale"""
        sql_upper = sql.upper()
        
        if "FROM" in sql_upper:
            from_pos = sql_upper.find("FROM")
            from_part = sql[from_pos + 4:].strip()
            
            # Prendre le premier mot après FROM
            words = from_part.split()
            if words:
                table = words[0].strip("`\"'")
                return table
        
        return None
    
    def _simplify_subqueries(self, sql: str) -> str:
        """Tente de simplifier les sous-requêtes"""
        # Pour l'instant, retourne la requête avec un commentaire
        return sql + "\n-- CONSIDÉRER: Remplacer les sous-requêtes par des JOIN ou des CTE"
    
# ==================== MODÈLES RL SIMPLIFIÉS ====================
class RLSimulator:
    """Simulateur d'agent RL pour l'optimisation d'index"""
    
    def __init__(self):
        print("🤖 Agent RL initialisé (mode simulation)")
        self.current_indexes = 2  # Nombre d'index simulé
        self.performance_history = []
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.max_indexes = RL_CONFIG['max_indexes']
    
    def get_status(self):
        """État actuel de la base"""
        return {
            'performance': 0.035 + np.random.uniform(-0.01, 0.01),  # Temps de requête simulé
            'index_count': self.current_indexes,
            'max_indexes': self.max_indexes,
            'status': 'active',
            'message': f'Base optimisée ({self.current_indexes}/{self.max_indexes} index)',
            'agent_mode': 'RL Simulation'
        }
    
    def optimize(self, steps=5, strategy='balanced'):
        """Exécute l'optimisation RL"""
        print(f"🔄 Optimisation RL demandée: {steps} étapes, stratégie: {strategy}")
        
        results = []
        
        for step in range(steps):
            # Logique de décision RL simplifiée
            if self.current_indexes < 2 and np.random.random() > 0.3:
                action = 'CREATE'
                self.current_indexes += 1
                reward = 0.8
            elif self.current_indexes > 3 and np.random.random() > 0.5:
                action = 'DROP'
                self.current_indexes -= 1
                reward = 0.4
            else:
                action = 'NOOP'
                reward = 0.1
            
            # Garder dans les limites
            self.current_indexes = max(1, min(self.current_indexes, self.max_indexes))
            
            # Performance simulée (s'améliore avec plus d'index jusqu'à un point)
            base_performance = 0.05
            index_benefit = min(self.current_indexes / self.max_indexes, 0.7)
            query_time = base_performance * (1 - index_benefit) + np.random.uniform(-0.005, 0.005)
            query_time = max(0.025, min(query_time, 0.1))
            
            step_result = {
                'step': step + 1,
                'action': action,
                'reward': round(reward, 4),
                'indexes': self.current_indexes,
                'query_time': round(query_time, 4),
                'explanation': self._get_action_explanation(action)
            }
            
            results.append(step_result)
            
            # Enregistrer dans l'historique
            self.performance_history.append({
                'step': step + 1,
                'query_time': query_time,
                'indexes': self.current_indexes,
                'reward': reward
            })
        
        return {
            'status': 'success',
            'steps': steps,
            'final_indexes': self.current_indexes,
            'final_query_time': results[-1]['query_time'] if results else 0.035,
            'total_reward': sum(r['reward'] for r in results),
            'steps_details': results,
            'strategy_used': strategy,
            'max_indexes': self.max_indexes
        }
    
    def get_recommendations(self):
        """Recommandations d'index basées sur l'apprentissage"""
        current_idx = self.current_indexes
        
        recommendations = []
        
        # Recommandations basées sur l'état actuel
        if current_idx < 2:
            recommendations.append({
                'type': 'CREATE',
                'priority': 'high',
                'description': 'Créer un index composite sur orders(client_id, orderdate)',
                'sql': 'CREATE INDEX idx_orders_client_date ON orders(client_id, orderdate)',
                'impact': 'Amélioration estimée: 40-60% sur les requêtes de jointure',
                'confidence': 0.85
            })
            
            recommendations.append({
                'type': 'CREATE',
                'priority': 'medium',
                'description': 'Index sur clients(wilaya) pour les filtres géographiques',
                'sql': 'CREATE INDEX idx_clients_wilaya ON clients(wilaya)',
                'impact': 'Accélération des requêtes par région',
                'confidence': 0.75
            })
        
        elif current_idx >= 4:
            recommendations.append({
                'type': 'DROP',
                'priority': 'medium',
                'description': 'Réduire le nombre d\'index pour optimiser les écritures',
                'sql': 'DROP INDEX idx_orders_test ON orders',
                'impact': 'Amélioration des INSERT/UPDATE de 15-25%',
                'confidence': 0.65
            })
        
        # Recommandations toujours pertinentes
        recommendations.append({
            'type': 'ANALYZE',
            'priority': 'low',
            'description': 'Analyser l\'utilisation des index existants',
            'sql': 'SELECT * FROM information_schema.statistics WHERE table_schema = DATABASE()',
            'impact': 'Identification des index sous-utilisés',
            'confidence': 0.9
        })
        
        return {
            'current_indexes': current_idx,
            'max_indexes': self.max_indexes,
            'recommendations': recommendations,
            'performance_trend': 'improving' if len(self.performance_history) > 1 and 
                                 self.performance_history[-1]['query_time'] < self.performance_history[0]['query_time'] 
                                 else 'stable'
        }
    
    def _get_action_explanation(self, action):
        """Explication des actions RL"""
        explanations = {
            'CREATE': 'L\'agent RL a décidé de créer un index pour améliorer les performances de lecture',
            'DROP': 'L\'agent RL a décidé de supprimer un index pour réduire l\'overhead des écritures',
            'NOOP': 'L\'agent RL a décidé de ne rien changer (état optimal atteint)'
        }
        return explanations.get(action, 'Action non expliquée')
    
    def get_learning_stats(self):
        """Statistiques d'apprentissage de l'agent RL"""
        if len(self.performance_history) < 2:
            return {
                'total_steps': len(self.performance_history),
                'average_reward': 0,
                'performance_improvement': 0,
                'learning_progress': 0
            }
        
        rewards = [h['reward'] for h in self.performance_history]
        query_times = [h['query_time'] for h in self.performance_history]
        
        return {
            'total_steps': len(self.performance_history),
            'average_reward': round(np.mean(rewards), 4),
            'performance_improvement': round((query_times[0] - query_times[-1]) / query_times[0] * 100, 1),
            'learning_progress': min(len(self.performance_history) / 100, 1.0),  # Progression simulée
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }

# ==================== MODÈLES ====================
class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"
    analyze_type: str = "full"

class SQLAnalysisRequest(BaseModel):
    sql: str
    analyze_type: str = "full"

class OptimizeRequest(BaseModel):
    steps: int = 5
    strategy: str = "balanced"

class RLTrainingRequest(BaseModel):
    episodes: int = 10
    learning_rate: float = 0.01
    exploration_rate: float = 0.3

# ==================== UTILS MySQL ====================
def get_db_connection():
    """Connexion à MySQL"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Erreur connexion MySQL: {err}")
        raise HTTPException(status_code=500, detail=f"Erreur connexion MySQL: {err}")

def execute_query(sql: str):
    """Exécute une requête SQL et retourne les résultats"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        start_time = time.time()
        cursor.execute(sql)
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "results": results,
            "execution_time": execution_time,
            "row_count": len(results),
            "is_slow": execution_time > SLOW_QUERY_THRESHOLD
        }
    except mysql.connector.Error as err:
        return {
            "success": False,
            "error": str(err),
            "execution_time": 0,
            "row_count": 0,
            "is_slow": False
        }
    finally:
        cursor.close()
        conn.close()

def get_table_info():
    """Récupère les informations des tables"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    tables_info = {}
    try:
        cursor.execute("SHOW TABLES")
        tables = [row[f'Tables_in_{MYSQL_CONFIG["database"]}'] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count_result = cursor.fetchone()
            count = count_result['count'] if count_result else 0
            
            cursor.execute(f"""
                SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE, INDEX_TYPE
                FROM INFORMATION_SCHEMA.STATISTICS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = '{table}'
                ORDER BY INDEX_NAME, SEQ_IN_INDEX
            """)
            indexes = cursor.fetchall()
            
            tables_info[table] = {
                "columns": columns,
                "row_count": count,
                "indexes": indexes,
                "index_count": len(indexes)
            }
    except mysql.connector.Error as err:
        print(f"Erreur récupération tables: {err}")
    finally:
        cursor.close()
        conn.close()
    
    return tables_info

# ==================== SIMULATION XGBOOST ====================
class XGBoostSimulator:
    """Simulateur de modèle XGBoost pour prédire si une requête est lente"""
    
    def __init__(self):
        print("🔮 Simulateur XGBoost initialisé")
    
    def predict(self, sql: str):
        """Prédit si une requête sera lente basé sur l'analyse du SQL"""
        sql_upper = sql.upper()
        
        # Facteurs influençant la lenteur
        complexity_score = 0
        reasons = []
        
        # Facteurs de complexité
        if "SELECT *" in sql_upper:
            complexity_score += 30
            reasons.append("Utilise SELECT *")
        
        join_count = sql_upper.count("JOIN")
        if join_count > 0:
            complexity_score += join_count * 15
            reasons.append(f"{join_count} jointure(s)")
        
        if "GROUP BY" in sql_upper:
            complexity_score += 20
            reasons.append("GROUP BY détecté")
        
        if sql_upper.count("SELECT") > 1:
            complexity_score += 25
            reasons.append("Sous-requêtes détectées")
        
        # Calcul de la probabilité
        base_probability = 0.5
        complexity_factor = min(complexity_score / 100, 0.5)
        slow_probability = base_probability + complexity_factor
        slow_probability = min(slow_probability, 0.95)
        
        # Détermination
        is_slow_predicted = slow_probability > 0.6
        confidence = abs(slow_probability - 0.5) * 2
        
        return {
            "is_slow": is_slow_predicted,
            "slow_probability": round(slow_probability, 3),
            "fast_probability": round(1 - slow_probability, 3),
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "complexity_score": complexity_score,
            "threshold": SLOW_QUERY_THRESHOLD
        }

# ==================== AGENT SADOP AVEC RL ====================
class SADOPAgent:
    """Agent SADOP complet avec XGBoost et RL"""
    def optimize_sql_query(self, sql: str):
        """Optimise une requête SQL lente"""
        
        # Analyser d'abord avec XGBoost
        xgb_prediction = self.xgb_predictor.predict(sql)
        
        # Optimiser la requête
        optimization_result = self.sql_optimizer.optimize_query(sql)
        
        # Construction de la réponse
        response = [
            "## 🔧 Optimisation de requête SQL",
            "",
            "### 📝 Requête originale:",
            f"```sql\n{sql}\n```",
            ""
        ]
        
        # Prédiction XGBoost
        response.append("### 🤖 Diagnostic XGBoost:")
        if xgb_prediction["is_slow"]:
            response.append(f"⚠️ **REQUÊTE LENTE DÉTECTÉE** (seuil: {xgb_prediction['threshold']}s)")
            response.append(f"- Probabilité lente: {xgb_prediction['slow_probability']*100:.1f}%")
            response.append(f"- Score complexité: {xgb_prediction['complexity_score']}/100")
        else:
            response.append(f"✅ **REQUÊTE RAPIDE**")
        
        # Version optimisée
        response.append("")
        response.append("### 🚀 Version optimisée:")
        response.append(f"```sql\n{optimization_result['optimized_sql']}\n```")
        
        # Amélioration estimée
        improvement = optimization_result['estimated_improvement']
        response.append("")
        response.append(f"### 📈 Amélioration estimée: **{improvement['estimated_percent']}%**")
        response.append(f"- Confiance: {improvement['confidence']*100:.0f}%")
        response.append(f"- Réduction temps: {improvement['expected_time_reduction']}")
        
        # Optimisations appliquées
        if optimization_result['optimizations']:
            response.append("")
            response.append("### 🔧 Optimisations appliquées:")
            
            for i, opt in enumerate(optimization_result['optimizations'], 1):
                response.append(f"**{i}. {opt['type'].replace('_', ' ').title()}**")
                response.append(f"- {opt['description']}")
                response.append(f"- Impact: {opt['impact']}")
                if opt.get('sql_example'):
                    response.append(f"```sql\n{opt['sql_example']}\n```")
                response.append("")
        
        # Recommandations RL
        response.append("")
        response.append("### 🤖 Recommandations RL pour prévenir:")
        
        rl_recommendations = self.rl_agent.get_recommendations()
        for i, rec in enumerate(rl_recommendations['recommendations'][:2], 1):
            response.append(f"{i}. **{rec['description']}**")
            response.append(f"   ```sql\n   {rec['sql']}\n   ```")
        
        # Test de performance
        response.append("")
        response.append("### ⚡ Test de performance:")
        response.append("Pour comparer les performances:")
        response.append(f"```sql\n-- Original:\nEXPLAIN {sql}\n\n-- Optimisé:\nEXPLAIN {optimization_result['optimized_sql']}\n```")
        
        return "\n".join(response)
    def __init__(self):
        print("🤖 Agent SADOP complet initialisé")
        self.tables = ["wilayas", "promotions", "products", "orders", "offers", "clients", "cart", "admin"]
        self.xgb_predictor = XGBoostSimulator()
        self.rl_agent = RLSimulator()
        self.sql_optimizer = SQLOptimizer()
        self.conversation_history = []
    
    async def query(self, user_input: str):
        """Traite une question utilisateur"""
        input_lower = user_input.lower()
        
        # Ajouter à l'historique
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": user_input
        })
        
        # Détection du type de question
        if any(keyword in input_lower for keyword in ["optimiser", "améliorer", "rendre plus rapide", "accélérer"]) and any(keyword in input_lower for keyword in ["select", "requête", "sql"]):
            # Extraire la requête SQL du message
            sql_query = self._extract_sql_from_message(user_input)
            if sql_query:
                return self.optimize_sql_query(sql_query)
            else:
                return "Je peux optimiser vos requêtes SQL. Veuillez fournir une requête SQL à optimiser."
        
        elif any(keyword in input_lower for keyword in ["select", "insert", "update", "delete", "from ", "where "]):
            return await self.analyze_sql(user_input)
        
        elif any(keyword in input_lower for keyword in ["optimiser", "optimize", "rl", "reinforcement"]):
            return self.handle_rl_request(user_input)
        
        elif any(keyword in input_lower for keyword in ["index", "indexes", "indexation"]):
            return self.get_index_recommendations()
        
        elif any(keyword in input_lower for keyword in ["statistique", "data", "données", "performance"]):
            return self.get_performance_analysis()
        
        elif any(keyword in input_lower for keyword in ["table", "tables", "schema", "structure"]):
            return self.get_database_info()
        
        else:
            return self.general_response(user_input)
    
    def _extract_sql_from_message(self, message: str) -> Optional[str]:
        """Extrait une requête SQL d'un message"""
        # Recherche de code SQL entre backticks
        import re
        
        # Pattern pour trouver du code SQL
        sql_patterns = [
            r"```sql\n(.*?)\n```",  # Code SQL avec backticks
            r"```\n(.*?)\n```",     # Code avec backticks
            r"'(.*?)'",             # Entre simples quotes
            r'"([^"]*SELECT[^"]*)"' # Entre doubles quotes avec SELECT
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, message, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Si pas trouvé, essayer de trouver une requête SQL dans le texte
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE"]
        for keyword in sql_keywords:
            if keyword in message.upper():
                # Prendre la ligne contenant le mot-clé
                lines = message.split('\n')
                for line in lines:
                    if keyword in line.upper():
                        return line.strip()
        
        return None
    
    async def analyze_sql(self, sql: str):
        """Analyse une requête SQL avec XGBoost"""
        # Prédiction XGBoost
        xgb_prediction = self.xgb_predictor.predict(sql)
        
        # Exécution réelle
        execution_result = execute_query(sql)
        
        # Construction de la réponse
        response = [
            "## 🔍 Analyse SQL SADOP",
            "",
            "### 📝 Requête analysée",
            f"```sql\n{sql}\n```",
            ""
        ]
        
        # Prédiction XGBoost
        response.append("### 🤖 Prédiction XGBoost")
        if xgb_prediction["is_slow"]:
            response.append(f"⚠️ **PRÉDIT COMME LENTE** (seuil: {SLOW_QUERY_THRESHOLD}s)")
            response.append(f"- Confiance: {xgb_prediction['confidence']*100:.1f}%")
            response.append(f"- Probabilité lente: {xgb_prediction['slow_probability']*100:.1f}%")
            
            if xgb_prediction["reasons"]:
                response.append("")
                response.append("**Raisons:**")
                for reason in xgb_prediction["reasons"]:
                    response.append(f"- {reason}")
        else:
            response.append(f"✅ **PRÉDIT COMME RAPIDE**")
            response.append(f"- Confiance: {xgb_prediction['confidence']*100:.1f}%")
        
        # Résultats d'exécution
        response.append("")
        response.append("### ⚡ Exécution réelle")
        if execution_result["success"]:
            response.append(f"- Temps: {execution_result['execution_time']:.3f}s")
            response.append(f"- Lignes: {execution_result['row_count']}")
            
            if execution_result["is_slow"]:
                response.append(f"🔴 **EXÉCUTION LENTE** (> {SLOW_QUERY_THRESHOLD}s)")
            else:
                response.append(f"🟢 **EXÉCUTION RAPIDE**")
        else:
            response.append(f"❌ Erreur: {execution_result.get('error', 'Inconnue')}")
        
        # Recommandations RL
        rl_status = self.rl_agent.get_status()
        response.append("")
        response.append("### 🤖 Recommandations RL")
        response.append(f"État actuel: {rl_status['index_count']}/{rl_status['max_indexes']} index")
        response.append(f"Performance: {rl_status['performance']:.3f}s")
        
        # Suggestions spécifiques
        if xgb_prediction["is_slow"] or execution_result.get("is_slow", False):
            response.append("")
            response.append("**💡 Actions recommandées:**")
            response.append("1. Utiliser l'optimisation RL pour améliorer les index")
            response.append("2. Exécuter `EXPLAIN` pour analyser le plan")
            response.append("3. Considérer l'ajout d'index sur les colonnes filtrées")
        
        return "\n".join(response)
    
    def handle_rl_request(self, request: str):
        """Gère les demandes liées au RL"""
        request_lower = request.lower()
        
        if "optimiser" in request_lower or "lancer" in request_lower:
            # Démarrer l'optimisation RL
            result = self.rl_agent.optimize(steps=5)
            
            response = [
                "## 🚀 Optimisation RL exécutée",
                "",
                f"**Stratégie:** {result['strategy_used']}",
                f"**Étapes:** {result['steps']}",
                f"**Index finaux:** {result['final_indexes']}/{RL_CONFIG['max_indexes']}",
                f"**Performance finale:** {result['final_query_time']:.3f}s",
                f"**Récompense totale:** {result['total_reward']:.3f}",
                "",
                "### 📊 Détails des étapes:"
            ]
            
            for step in result['steps_details']:
                response.append(f"**Étape {step['step']}:** {step['action']}")
                response.append(f"  - Index: {step['indexes']} | Temps: {step['query_time']:.3f}s | Reward: {step['reward']:.3f}")
                response.append(f"  - {step['explanation']}")
                response.append("")
            
            return "\n".join(response)
        
        elif "statut" in request_lower or "état" in request_lower:
            # Statut RL
            status = self.rl_agent.get_status()
            stats = self.rl_agent.get_learning_stats()
            
            response = [
                "## 📊 État de l'agent RL",
                "",
                f"**Mode:** {status['agent_mode']}",
                f"**Index actuels:** {status['index_count']}/{status['max_indexes']}",
                f"**Performance:** {status['performance']:.3f}s",
                f"**Statut:** {status['status']}",
                "",
                "### 📈 Statistiques d'apprentissage:",
                f"- Étapes totales: {stats['total_steps']}",
                f"- Récompense moyenne: {stats['average_reward']:.3f}",
                f"- Amélioration performance: {stats['performance_improvement']}%",
                f"- Progression apprentissage: {stats['learning_progress']*100:.1f}%",
                "",
                "**Message:** " + status['message']
            ]
            
            return "\n".join(response)
        
        else:
            return self.general_rl_info()
    
    def get_index_recommendations(self):
        """Retourne les recommandations d'index du RL"""
        recs = self.rl_agent.get_recommendations()
        
        response = [
            "## 🗂️ Recommandations d'index (RL)",
            "",
            f"**Index actuels:** {recs['current_indexes']}/{recs['max_indexes']}",
            f"**Tendance performance:** {recs['performance_trend']}",
            "",
            "### 🎯 Recommandations prioritaires:"
        ]
        
        for i, rec in enumerate(recs['recommendations'], 1):
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            response.append(f"{priority_icon} **{i}. [{rec['type']}] {rec['description']}**")
            response.append(f"   ```sql\n   {rec['sql']}\n   ```")
            response.append(f"   Impact: {rec['impact']} (confiance: {rec['confidence']*100:.0f}%)")
            response.append("")
        
        return "\n".join(response)
    
    def get_performance_analysis(self):
        """Analyse de performance complète"""
        # Récupérer les informations
        rl_status = self.rl_agent.get_status()
        learning_stats = self.rl_agent.get_learning_stats()
        
        response = [
            "## 📊 Analyse de performance SADOP",
            "",
            "### 🤖 État de l'agent RL:",
            f"- Index: {rl_status['index_count']}/{rl_status['max_indexes']}",
            f"- Performance: {rl_status['performance']:.3f}s",
            f"- Statut: {rl_status['status']}",
            "",
            "### 📈 Apprentissage RL:",
            f"- Étapes: {learning_stats['total_steps']}",
            f"- Amélioration: {learning_stats['performance_improvement']}%",
            f"- Récompense moyenne: {learning_stats['average_reward']:.3f}",
            "",
            "### ⚡ Seuils de performance:",
            f"- Requête lente: > {SLOW_QUERY_THRESHOLD}s",
            f"- Objectif RL: < {SLOW_QUERY_THRESHOLD * 0.7:.3f}s",
            "",
            "### 🎯 Recommandations:"
        ]
        
        # Recommandations basées sur l'état
        if rl_status['index_count'] < 2:
            response.append("1. **Lancer l'optimisation RL** pour ajouter des index stratégiques")
            response.append("2. **Créer un index composite** sur orders(client_id, orderdate)")
        elif rl_status['performance'] > SLOW_QUERY_THRESHOLD:
            response.append("1. **Analyser les requêtes lentes** avec XGBoost")
            response.append("2. **Ajuster la stratégie RL** pour optimisation aggressive")
        else:
            response.append("1. **Maintenir l'état actuel** - performance optimale atteinte")
            response.append("2. **Surveiller** les nouvelles requêtes")
        
        return "\n".join(response)
    
    def get_database_info(self):
        """Informations sur la base de données"""
        try:
            tables_info = get_table_info()
            
            response = [
                "## 🗃️ Structure de la base POS",
                "",
                "### 📊 Vue d'ensemble:",
                f"- Tables: {len(tables_info)}",
                "- Principales: orders, clients, products, cart",
                "",
                "### 🔗 Relations clés:",
                "- orders.client_id → clients.id",
                "- orders.wilaya_id → wilayas.id",
                "- cart.product_id → products.id",
                "",
                "### 🗂️ Statistiques d'index (simulé):"
            ]
            
            # Statistiques d'index simulées
            table_stats = [
                ("orders", 15642, 3),
                ("clients", 5231, 2),
                ("products", 1250, 2),
                ("cart", 8923, 1)
            ]
            
            for table, rows, indexes in table_stats:
                response.append(f"- **{table}**: {rows:,} lignes, {indexes} index")
            
            return "\n".join(response)
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
    def general_response(self, question: str):
        """Réponse générale"""
        return f"""
## 🤖 Assistant SADOP - Système Complet

**Votre question:** {question}

**Fonctionnalités disponibles:**

### 🔍 **Diagnostic XGBoost:**
- Prédire si une requête sera lente (seuil: {SLOW_QUERY_THRESHOLD}s)
- Analyser la complexité des requêtes SQL
- Identifier les problèmes de performance

### 🤖 **Optimisation RL:**
- Apprentissage par renforcement pour les index
- Optimisation automatique de la base
- Recommandations intelligentes d'index

### 📊 **Analyse:**
- Statistiques de performance
- Structure de la base de données
- Historique d'optimisation

### 💡 **Commandes utiles:**
- "Analyser cette requête: SELECT * FROM orders"
- "Lancer l'optimisation RL"
- "Voir les recommandations d'index"
- "Statut de l'agent RL"
- "Performance de la base"
"""

    def general_rl_info(self):
        """Informations générales sur le RL"""
        return """
## 🤖 Apprentissage par Renforcement (RL)

**Comment fonctionne l'agent RL SADOP:**

### 🎯 **Objectif:**
Optimiser automatiquement les index MySQL pour:
- Accélérer les requêtes de lecture
- Minimiser l'impact sur les écritures
- Trouver le compromis optimal

### 🔄 **Processus:**
1. **Observation:** État actuel de la base (index, performance)
2. **Action:** Créer, supprimer ou maintenir des index
3. **Récompense:** Amélioration de performance - coût de l'action
4. **Apprentissage:** Ajustement de la stratégie

### ⚡ **Actions possibles:**
- **CREATE INDEX:** Améliore les lectures mais coûte des ressources
- **DROP INDEX:** Libère des ressources mais peut ralentir les lectures
- **NOOP:** Maintenir l'état actuel

### 🚀 **Comment l'utiliser:**
- "Lancer l'optimisation RL"
- "Voir le statut du RL"
- "Optimiser avec stratégie aggressive"
- "Recommandations d'index RL"
"""

# ==================== INITIALISATION ====================
app = FastAPI(
    title="SADOP API - Système Complet",
    description="Système Autonome de Diagnostic et d'Optimisation avec XGBoost et RL",
    version="4.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance de l'agent
sadop_agent = SADOPAgent()

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "service": "SADOP API v4.0",
        "description": "Système complet avec XGBoost et Reinforcement Learning",
        "author": "APBD Team",
        "version": "4.0",
        "features": {
            "xgboost": "Prédiction de requêtes lentes",
            "rl": "Optimisation automatique d'index",
            "sql_analysis": "Analyse de requêtes SQL",
            "performance_monitoring": "Surveillance des performances"
        },
        "endpoints": {
            "GET /": "Cette page",
            "GET /health": "État du service",
            "GET /api/rl/status": "Statut de l'agent RL",
            "GET /api/rl/recommendations": "Recommandations RL",
            "POST /api/rl/optimize": "Lancer l'optimisation RL",
            "POST /api/analyze/sql": "Analyse de requête SQL",
            "POST /chat": "Chat avec l'agent IA"
        }
    }

@app.get("/health")
async def health_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        db_name = "unknown"
    
    # Statut RL
    rl_status = sadop_agent.rl_agent.get_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "database_name": db_name,
        "xgboost": "ready",
        "rl_agent": rl_status['status'],
        "rl_indexes": f"{rl_status['index_count']}/{rl_status['max_indexes']}",
        "version": "4.0",
        "slow_query_threshold": SLOW_QUERY_THRESHOLD
    }

# ==================== ENDPOINTS RL ====================

@app.get("/api/rl/status")
async def get_rl_status():
    """Statut de l'agent RL"""
    try:
        status = sadop_agent.rl_agent.get_status()
        stats = sadop_agent.rl_agent.get_learning_stats()
        
        return {
            "success": True,
            "data": {
                "status": status,
                "learning_stats": stats,
                "configuration": RL_CONFIG
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/rl/recommendations")
async def get_rl_recommendations():
    """Recommandations d'index du RL"""
    try:
        recommendations = sadop_agent.rl_agent.get_recommendations()
        return {"success": True, "data": recommendations}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/rl/optimize")
async def optimize_with_rl(request: OptimizeRequest):
    """Lance l'optimisation RL"""
    try:
        result = sadop_agent.rl_agent.optimize(
            steps=request.steps,
            strategy=request.strategy
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Optimisation RL terminée ({request.steps} étapes)"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/rl/learning-stats")
async def get_rl_learning_stats():
    """Statistiques d'apprentissage du RL"""
    try:
        stats = sadop_agent.rl_agent.get_learning_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== ENDPOINTS XGBOOST ====================

@app.post("/api/analyze/sql")
async def analyze_sql_endpoint(request: SQLAnalysisRequest):
    """Analyse de requête SQL avec XGBoost"""
    try:
        # Prédiction XGBoost
        xgb_prediction = sadop_agent.xgb_predictor.predict(request.sql)
        
        # Exécution réelle
        execution_result = execute_query(request.sql)
        
        return {
            "success": True,
            "sql": request.sql,
            "prediction": xgb_prediction,
            "execution": execution_result,
            "slow_threshold": SLOW_QUERY_THRESHOLD
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== ENDPOINTS CHAT ====================

@app.post("/chat")
async def chat_with_agent(request: QueryRequest):
    """Chat avec l'agent SADOP complet"""
    try:
        print(f"📩 Question: {request.question[:100]}...")
        
        response = await sadop_agent.query(request.question)
        
        return {
            "success": True,
            "question": request.question,
            "response": response,
            "source": "SADOP Agent (XGBoost + RL)",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== ENDPOINTS SUPPLEMENTAIRES ====================

@app.get("/api/tables")
async def get_tables():
    """Liste des tables"""
    try:
        tables_info = get_table_info()
        return {
            "success": True,
            "data": tables_info,
            "count": len(tables_info)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/performance")
async def get_performance_summary():
    """Résumé des performances"""
    try:
        rl_status = sadop_agent.rl_agent.get_status()
        learning_stats = sadop_agent.rl_agent.get_learning_stats()
        
        return {
            "success": True,
            "data": {
                "rl_status": rl_status,
                "learning_stats": learning_stats,
                "slow_threshold": SLOW_QUERY_THRESHOLD,
                "max_indexes": RL_CONFIG['max_indexes']
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/optimize/sql")
async def optimize_sql_endpoint(request: SQLAnalysisRequest):
    """Optimise une requête SQL"""
    try:
        optimization_result = sadop_agent.sql_optimizer.optimize_query(request.sql)
        
        # Prédiction XGBoost
        xgb_prediction = sadop_agent.xgb_predictor.predict(request.sql)
        
        return {
            "success": True,
            "original_sql": request.sql,
            "optimized_sql": optimization_result["optimized_sql"],
            "optimizations": optimization_result["optimizations"],
            "improvement": optimization_result["estimated_improvement"],
            "xgboost_prediction": xgb_prediction
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
# ==================== LANCEMENT ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SADOP API v4.0 - Système Complet (XGBoost + RL)")
    print("=" * 60)
    print(f"🔮 XGBoost: Prédiction requêtes lentes (> {SLOW_QUERY_THRESHOLD}s)")
    print(f"🤖 RL: Optimisation automatique d'index")
    print(f"📊 Max index: {RL_CONFIG['max_indexes']}")
    print(f"🌐 URL: http://localhost:8000")
    print(f"📚 Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")