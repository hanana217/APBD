# agent/langchain_agent.py - VERSION AMÉLIORÉE
import asyncio
import sys
import os

class SADOPAgent:
    """Agent simple pour l'analyse de requêtes SQL"""

    def __init__(self):
        print("✅ Agent SADOP initialisé (mode local)")
        
        # Essayer de charger XGBoost
        self.xgboost_model = None
        self.load_xgboost()

    def load_xgboost(self):
        """Tente de charger le modèle XGBoost depuis backend"""
        try:
            # Calculer le chemin vers backend
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.join(current_dir, "..", "apbd_interface", "backend")
            
            if not os.path.exists(backend_dir):
                print(f"⚠️ Dossier backend non trouvé: {backend_dir}")
                return
            
            # Ajouter au path
            sys.path.append(backend_dir)
            
            # Importer
            from xgboost_api import xgboost_model
            self.xgboost_model = xgboost_model
            print("✅ Modèle XGBoost chargé")
            
        except ImportError as e:
            print(f"⚠️ Impossible de charger XGBoost: {e}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de XGBoost: {e}")

    def analyze_sql(self, sql_query: str):
        """Analyse détaillée d'une requête SQL"""
        sql_upper = sql_query.upper()
        observations = []
        recommendations = []
        performance = "⚡ Rapide"
        severity = "🟢"

        # Détection des problèmes
        if "SELECT *" in sql_upper:
            observations.append("Utilise SELECT * (récupère toutes les colonnes)")
            recommendations.append("Spécifiez uniquement les colonnes nécessaires")
            severity = "🟡"

        if "WHERE" not in sql_upper and "LIMIT" not in sql_upper:
            observations.append("Pas de filtre WHERE ou LIMIT")
            recommendations.append("Ajoutez WHERE pour réduire le volume")
            performance = "⚠️ Potentiellement lent"
            severity = "🟡"

        if "JOIN" in sql_upper:
            observations.append("Contient des jointures")
            recommendations.append("Vérifiez les index sur les colonnes de jointure")
            severity = "🟡"

        if "LIKE '%" in sql_upper:
            observations.append("Utilise LIKE avec wildcard au début")
            recommendations.append("Évitez 'LIKE %...' (pas d'index possible)")
            performance = "🐌 Très lent"
            severity = "🔴"

        if "ORDER BY" in sql_upper:
            observations.append("Contient ORDER BY")
            recommendations.append("Indexez les colonnes de tri")

        if "GROUP BY" in sql_upper:
            observations.append("Contient GROUP BY")
            recommendations.append("Indexez les colonnes de regroupement")

        if "DISTINCT" in sql_upper:
            observations.append("Utilise DISTINCT")
            recommendations.append("Considérez une optimisation avec GROUP BY")

        if not observations:
            observations.append("Requête SQL bien formée")
            performance = "🚀 Très rapide"
            severity = "🟢"

        if not recommendations:
            recommendations.append("Considérez l'ajout d'index stratégiques")

        # Prédiction XGBoost si disponible
        xgboost_pred = None
        if self.xgboost_model:
            try:
                xgboost_pred = self.xgboost_model.predict(sql_query)
                if xgboost_pred:
                    if xgboost_pred['is_slow']:
                        performance = f"🐌 Lente (XGBoost: {xgboost_pred['confidence']:.1%})"
                        severity = "🔴"
                    else:
                        performance = f"⚡ Rapide (XGBoost: {xgboost_pred['confidence']:.1%})"
            except:
                pass

        return {
            "observations": observations,
            "recommendations": recommendations,
            "performance": performance,
            "severity": severity,
            "xgboost": xgboost_pred
        }

    async def query(self, user_input: str):
        """Traite une question utilisateur"""
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE"]
        
        # Vérifier si c'est une requête SQL
        is_sql = any(k in user_input.upper() for k in sql_keywords)
        
        if not is_sql:
            return f"""
**Question :** {user_input}

**Réponse :** Je suis l'agent SADOP pour l'optimisation SQL.
Posez-moi une requête SQL à analyser, par exemple :
- `SELECT * FROM users WHERE age > 30`
- `SELECT id, name FROM products ORDER BY price DESC`
- `UPDATE orders SET status = 'completed' WHERE id = 123`

**Ou demandez-moi :**
- Comment optimiser cette requête ?
- Faut-il un index sur cette colonne ?
- Pourquoi cette requête est lente ?
"""

        # Analyser la requête SQL
        analysis = self.analyze_sql(user_input)
        
        # Construire la réponse
        response = [
            "## 🔍 Analyse SQL SADOP",
            "",
            "### 📝 Requête analysée",
            "```sql",
            user_input,
            "```",
            "",
            f"### 📊 Performance : {analysis['severity']} {analysis['performance']}",
            "",
            "### 👁️ Observations",
        ]
        
        for obs in analysis["observations"]:
            response.append(f"- {obs}")
        
        response.append("")
        response.append("### 💡 Recommandations")
        
        for rec in analysis["recommendations"]:
            response.append(f"- {rec}")
        
        # Ajouter des suggestions d'index si pertinent
        if any(word in user_input.upper() for word in ["WHERE", "JOIN", "ORDER BY", "GROUP BY"]):
            response.append("")
            response.append("### 🗂️ Suggestions d'index")
            
            # Logique simple de suggestion d'index
            if "WHERE" in user_input.upper():
                response.append("- Indexez les colonnes dans la clause WHERE")
            if "JOIN" in user_input.upper():
                response.append("- Indexez les colonnes de jointure (ON ...)")
            if "ORDER BY" in user_input.upper():
                response.append("- Indexez les colonnes dans ORDER BY")
        
        # Ajouter les prédictions XGBoost si disponibles
        if analysis["xgboost"]:
            pred = analysis["xgboost"]
            response.append("")
            response.append("### 🤖 Prédiction XGBoost")
            response.append(f"- Probabilité lente : {pred.get('slow_probability', 0):.1%}")
            response.append(f"- Probabilité rapide : {pred.get('fast_probability', 0):.1%}")
            if 'features' in pred:
                response.append(f"- Lignes examinées estimées : {pred['features'].get('rows_examined', 'N/A')}")
        
        response.append("")
        response.append("### 🔧 Commande de diagnostic")
        response.append("```sql")
        response.append(f"EXPLAIN {user_input}")
        response.append("```")
        
        return "\n".join(response)


# Instance globale pour l'API
sadop_agent = SADOPAgent()

# -------------------------------------------------
# Test local
# -------------------------------------------------
if __name__ == "__main__":
    agent = SADOPAgent()
    
    tests = [
        "SELECT * FROM users",
        "SELECT id, name FROM products WHERE price > 100 ORDER BY created_at DESC",
        "SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE users.age > 30",
        "Comment optimiser ma base de données ?"
    ]
    
    for test in tests:
        print("\n" + "="*60)
        print(f"Test: {test}")
        print("="*60)
        result = asyncio.run(agent.query(test))
        print(result[:500] + "..." if len(result) > 500 else result)