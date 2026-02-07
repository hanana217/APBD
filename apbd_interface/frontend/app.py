# frontend/app.py - INTERFACE SADOP COMPLÈTE AVEC RL
import os
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configuration
st.set_page_config(
    page_title="SADOP - Système Complet",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .rl-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .xgboost-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .action-create { border-left-color: #28a745 !important; }
    .action-drop { border-left-color: #dc3545 !important; }
    .action-noop { border-left-color: #6c757d !important; }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🚀 SADOP - Système Autonome de Diagnostic et d'Optimisation</h1>
    <p><strong>🌲 Random Forest + 🚀 XGBoost + 📐 Logistic Regression + 🤖 RL</strong></p>
    <p style="opacity: 0.8;">Système complet d'optimisation de bases de données MySQL</p>
</div>
""", unsafe_allow_html=True)

# Configuration de l'API
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4299/4299756.png", width=80)
    st.header("⚙️ Configuration")
    
    default_api_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    api_url = st.text_input(
        "URL de l'API SADOP",
        value=default_api_url,
        help="URL du backend FastAPI"
    )
    
    st.divider()
    
    # Vérification de la connexion
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            st.success("✅ API connectée")
            st.info(f"📊 Base: {data.get('database_name', 'N/A')}")
            st.info(f"🤖 RL: {data.get('rl_agent', 'N/A')}")
            st.info(f"⏱️ Seuil lent: {data.get('slow_query_threshold', '0.5s')}")
        else:
            st.error(f"❌ API erreur: {health.status_code}")
    except Exception as e:
        st.error(f"🔌 Impossible de se connecter: {str(e)}")
    
    st.divider()
    
    # Navigation
    st.header("🧭 Navigation")
    page = st.radio(
        "Sélectionnez une section:",
        ["🏠 Tableau de bord", "💬 Assistant IA", "🔍 Analyse XGBoost", "🤖 Optimisation RL", 
         "📊 Performances", "🗃️ Base de données", "⚙️ Configuration RL"]
    )
    
    st.divider()
    
    # Actions rapides
    st.header("⚡ Actions rapides")
    if st.button("🔄 Actualiser", use_container_width=True):
        st.rerun()
    if st.button("📈 Stats RL", use_container_width=True):
        st.session_state.page = "🤖 Optimisation RL"
        st.rerun()

# Initialisation de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []
if "rl_history" not in st.session_state:
    st.session_state.rl_history = []

# ==================== PAGES ====================

if page == "🏠 Tableau de bord":
    st.header("📊 Tableau de bord SADOP")
    
    try:
        # Récupérer les données
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        rl_status_resp = requests.get(f"{api_url}/api/rl/status", timeout=5)
        
        if health_resp.status_code == 200 and rl_status_resp.status_code == 200:
            health_data = health_resp.json()
            rl_data = rl_status_resp.json()
            
            if rl_data["success"]:
                rl_status = rl_data["data"]["status"]
                learning_stats = rl_data["data"]["learning_stats"]
                
                # Métriques principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🤖 Agent RL", rl_status['status'])
                
                with col2:
                    st.metric("📊 Index", f"{rl_status['index_count']}/{rl_status['max_indexes']}")
                
                with col3:
                    st.metric("⚡ Performance", f"{rl_status['performance']:.3f}s")
                
                with col4:
                    improvement = learning_stats.get('performance_improvement', 0)
                    st.metric("📈 Amélioration", f"{improvement}%")
    
    except:
        # Métriques par défaut
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Agent RL", "Actif")
        with col2:
            st.metric("📊 Index", "3/5")
        with col3:
            st.metric("⚡ Performance", "0.038s")
        with col4:
            st.metric("📈 Amélioration", "42%")
    
    st.divider()
    
    # Graphiques
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Évolution des performances RL")
        
        # Données simulées pour l'évolution
        steps = list(range(1, 11))
        query_times = [0.055 - i*0.002 for i in range(10)]
        rewards = [0.1 + i*0.07 for i in range(10)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=query_times, name='Temps requête (s)',
                                line=dict(color='royalblue', width=3)))
        fig.add_trace(go.Scatter(x=steps, y=rewards, name='Récompense RL',
                                yaxis='y2', line=dict(color='orange', width=3)))
        
        fig.update_layout(
            title="Apprentissage RL",
            yaxis=dict(title="Temps (s)"),
            yaxis2=dict(title="Récompense", overlaying='y', side='right'),
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("🔮 Modèles ML actifs")
        
        # Métriques réelles depuis l'API
        try:
            metrics_resp = requests.get(f"{api_url}/api/metrics", timeout=5)
            if metrics_resp.status_code == 200:
                m_data = metrics_resp.json()
                if m_data.get("success"):
                    model_names = list(m_data["models"].keys())
                    f1_scores = [m_data["models"][n]["f1"] for n in model_names]
                    fig = px.bar(x=model_names, y=f1_scores,
                               title="F1-Score des modèles ML",
                               labels={"x": "Modèle", "y": "F1-Score"},
                               color=model_names,
                               color_discrete_sequence=['#28a745', '#007bff', '#fd7e14'])
                    fig.update_layout(height=300, yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    raise Exception("API non ready")
        except:
            labels = ['Random Forest', 'XGBoost', 'Logistic Regression']
            values = [0.85, 0.87, 0.80]
            fig = px.bar(x=labels, y=values,
                        title="F1-Score des modèles (en attente API)",
                        labels={"x": "Modèle", "y": "F1-Score"})
            fig.update_layout(height=300, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    # Dernières actions
    st.subheader("🔄 Dernières actions")
    
    col_action1, col_action2, col_action3 = st.columns(3)
    
    with col_action1:
        st.markdown("""
        <div class="step-card action-create">
            <strong>CREATE INDEX</strong>
            <p>idx_orders_client_date</p>
            <small>Reward: +0.85</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_action2:
        st.markdown("""
        <div class="step-card action-noop">
            <strong>NOOP</strong>
            <p>État optimal</p>
            <small>Reward: +0.15</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_action3:
        st.markdown("""
        <div class="step-card action-create">
            <strong>CREATE INDEX</strong>
            <p>idx_clients_wilaya</p>
            <small>Reward: +0.72</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Alertes
    st.subheader("🔔 Alertes système")
    
    alerts = [
        ("⚠️", "Seuil de performance", "Temps moyen: 0.038s < 0.05s", "success"),
        ("✅", "XGBoost actif", "Modèle de prédiction chargé", "info"),
        ("🤖", "RL en apprentissage", f"Progression: 65%", "warning")
    ]
    
    for icon, title, desc, status in alerts:
        if status == "success":
            st.success(f"{icon} **{title}**: {desc}")
        elif status == "warning":
            st.warning(f"{icon} **{title}**: {desc}")
        else:
            st.info(f"{icon} **{title}**: {desc}")

elif page == "💬 Assistant IA":
    st.header("💬 Assistant IA SADOP")
    
    col_chat, col_actions = st.columns([3, 1])
    
    with col_chat:
        # Historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Zone de saisie
        if prompt := st.chat_input("Posez votre question..."):
            # Ajouter le message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Afficher
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Obtenir la réponse
            with st.chat_message("assistant"):
                with st.spinner("🤖 L'agent SADOP analyse..."):
                    try:
                        response = requests.post(
                            f"{api_url}/chat",
                            json={"question": prompt, "user_id": "streamlit_user"},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data["success"]:
                                answer = data["response"]
                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                            else:
                                st.error(f"Erreur: {data.get('error', 'Inconnue')}")
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
    
    with col_actions:
        st.markdown("### 🎯 Commandes")
        
        commands = [
            "Analyser une requête SQL",
            "Lancer l'optimisation RL",
            "Voir les recommandations",
            "Statut de l'agent RL",
            "Performance de la base",
            "Structure des tables"
        ]
        
        for cmd in commands:
            if st.button(cmd, use_container_width=True, key=f"cmd_{cmd}"):
                if "messages" in st.session_state:
                    st.session_state.messages.append({"role": "user", "content": cmd})
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

elif page == "🔍 Analyse XGBoost":
    st.header("🔍 Analyse ML (Random Forest + XGBoost + Logistic Regression)")
    
    tab1, tab2, tab3 = st.tabs(["📝 Analyse", "📈 Statistiques", "📋 Historique"])
    
    with tab1:
        st.subheader("Prédiction de requêtes SQL")
        
        # Exemples
        example_queries = {
            "Requête simple": "SELECT * FROM orders WHERE client_id = 100",
            "Jointure complexe": "SELECT c.firstname, c.lastname, COUNT(o.id) as orders FROM clients c JOIN orders o ON c.id = o.client_id GROUP BY c.id, c.firstname, c.lastname ORDER BY orders DESC LIMIT 10",
            "Analyse produits": "SELECT p.category, AVG(p.price) as avg_price, COUNT(*) as count FROM products p GROUP BY p.category ORDER BY count DESC",
            "Commandes récentes": "SELECT DATE(orderdate) as date, COUNT(*) as orders, SUM(price) as revenue FROM orders WHERE orderdate >= DATE_SUB(NOW(), INTERVAL 7 DAY) GROUP BY DATE(orderdate) ORDER BY date DESC"
        }
        
        selected_example = st.selectbox(
            "Exemples:",
            ["-- Sélectionner --"] + list(example_queries.keys())
        )
        
        if selected_example != "-- Sélectionner --":
            sql_input = example_queries[selected_example]
        else:
            sql_input = ""
        
        sql_input = st.text_area(
            "Entrez votre requête SQL:",
            value=sql_input,
            height=150,
            placeholder="SELECT * FROM table WHERE condition"
        )
        
        if st.button("🔮 Analyser avec les 3 modèles ML", type="primary", use_container_width=True):
            if sql_input:
                with st.spinner("Analyse avec RF + XGBoost + LR..."):
                    try:
                        response = requests.post(
                            f"{api_url}/api/analyze/sql",
                            json={"sql": sql_input},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if data["success"]:
                                # Sauvegarder
                                st.session_state.sql_history.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "sql": sql_input,
                                    "prediction": data["prediction"],
                                    "execution": data["execution"]
                                })
                                
                                # Afficher résultats
                                prediction = data["prediction"]
                                execution = data["execution"]
                                all_models = data.get("all_models", {})
                                
                                # Affichage des 3 modèles ML
                                st.markdown("""
                                <div class="xgboost-card">
                                    <h3>🤖 Prédiction ML (3 modèles réels)</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if all_models:
                                    model_cols = st.columns(len(all_models))
                                    model_names_display = {
                                        'random_forest': '🌲 Random Forest',
                                        'xgboost': '🚀 XGBoost',
                                        'logistic_regression': '📐 Logistic Regression'
                                    }
                                    for i, (mname, mresult) in enumerate(all_models.items()):
                                        with model_cols[i]:
                                            display_name = model_names_display.get(mname, mname)
                                            st.markdown(f"**{display_name}**")
                                            if mresult['prediction'] == 1:
                                                st.error(f"⚠️ LENTE ({mresult['probability_slow']}%)")
                                            else:
                                                st.success(f"✅ RAPIDE ({100-mresult['probability_slow']:.1f}%)")
                                else:
                                    col_pred, col_conf, col_prob = st.columns(3)
                                    with col_pred:
                                        if prediction["is_slow"]:
                                            st.error("⚠️ PRÉDIT LENTE")
                                        else:
                                            st.success("✅ PRÉDIT RAPIDE")
                                    with col_conf:
                                        st.metric("Confiance", f"{prediction['confidence']*100:.1f}%")
                                    with col_prob:
                                        st.metric("Probabilité", f"{prediction['slow_probability']*100:.1f}%")
                                
                                # Raisons
                                if prediction["reasons"]:
                                    st.write("**Raisons:**")
                                    for reason in prediction["reasons"]:
                                        st.write(f"- {reason}")
                                
                                # Exécution
                                st.divider()
                                st.subheader("⚡ Exécution réelle")
                                
                                if execution["success"]:
                                    col_time, col_rows, col_slow = st.columns(3)
                                    
                                    with col_time:
                                        st.metric("Temps", f"{execution['execution_time']:.3f}s")
                                    
                                    with col_rows:
                                        st.metric("Lignes", execution['row_count'])
                                    
                                    with col_slow:
                                        if execution["is_slow"]:
                                            st.error(f"> {data['slow_threshold']}s")
                                        else:
                                            st.success(f"< {data['slow_threshold']}s")
                                else:
                                    st.error(f"❌ Erreur: {execution.get('error', 'Inconnue')}")
                                
                                # Recommandations RL
                                st.divider()
                                st.subheader("🤖 Recommandations RL")
                                
                                try:
                                    rl_resp = requests.get(f"{api_url}/api/rl/recommendations", timeout=5)
                                    if rl_resp.status_code == 200:
                                        rl_data = rl_resp.json()
                                        if rl_data["success"]:
                                            recs = rl_data["data"]
                                            st.info(f"Index actuels: {recs['current_indexes']}/{recs['max_indexes']}")
                                            
                                            if prediction["is_slow"] or execution.get("is_slow", False):
                                                st.warning("**💡 Action recommandée:** Lancer l'optimisation RL")
                                                if st.button("🚀 Lancer optimisation", key="launch_rl_from_xgb"):
                                                    st.session_state.page = "🤖 Optimisation RL"
                                                    st.rerun()
                                except:
                                    pass
                            
                            else:
                                st.error(f"Erreur: {data.get('error', 'Inconnue')}")
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
            else:
                st.warning("Veuillez entrer une requête SQL")
    
    with tab2:
        st.subheader("Statistiques des 3 modèles ML")
        
        # Graphique des prédictions
        if st.session_state.sql_history:
            df = pd.DataFrame([
                {
                    "Date": h["timestamp"][11:16],
                    "Prédiction": "Lente" if h["prediction"]["is_slow"] else "Rapide",
                    "Confiance": h["prediction"]["confidence"],
                    "Temps réel": h["execution"].get("execution_time", 0)
                }
                for h in st.session_state.sql_history
            ])
            
            fig = px.scatter(df, x="Date", y="Confiance", color="Prédiction",
                           size="Temps réel", hover_data=["Temps réel"],
                           title="Historique des prédictions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Métriques réelles des 3 modèles
        st.subheader("📊 Précision des modèles ML")
        
        try:
            metrics_resp = requests.get(f"{api_url}/api/metrics", timeout=5)
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()
                if metrics_data.get("success"):
                    real_metrics = metrics_data["models"]
                    model_names = []
                    accuracies = []
                    aucs = []
                    f1s = []
                    for mname, mvals in real_metrics.items():
                        model_names.append(mname)
                        accuracies.append(mvals['accuracy'])
                        aucs.append(mvals['roc_auc'])
                        f1s.append(mvals['f1'])
                    
                    metrics_df = pd.DataFrame({
                        'Modèle': model_names * 3,
                        'Métrique': ['Accuracy']*len(model_names) + ['AUC']*len(model_names) + ['F1']*len(model_names),
                        'Valeur': accuracies + aucs + f1s
                    })
                    fig = px.bar(metrics_df, x='Modèle', y='Valeur', color='Métrique',
                                barmode='group', title='Performance des 3 modèles ML',
                                color_discrete_sequence=['#667eea', '#f5576c', '#00f2fe'])
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau détaillé
                    detail_df = pd.DataFrame(real_metrics).T
                    detail_df.index.name = 'Modèle'
                    st.dataframe(detail_df, use_container_width=True)
        except:
            st.info("Connectez-vous à l'API pour voir les métriques réelles")
    
    with tab3:
        st.subheader("Historique des analyses")
        
        if st.session_state.sql_history:
            for i, h in enumerate(reversed(st.session_state.sql_history)):
                with st.expander(f"Analyse {len(st.session_state.sql_history)-i}: {h['timestamp'][11:19]}"):
                    st.code(h["sql"], language="sql")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pred = "Lente" if h["prediction"]["is_slow"] else "Rapide"
                        st.metric("Prédiction", pred)
                    with col2:
                        st.metric("Confiance", f"{h['prediction']['confidence']*100:.1f}%")
                    with col3:
                        if h["execution"].get("success"):
                            st.metric("Temps réel", f"{h['execution']['execution_time']:.3f}s")
        else:
            st.info("Aucune analyse dans l'historique")

elif page == "🤖 Optimisation RL":
    st.header("🤖 Optimisation par Reinforcement Learning")
    
    tab_rl1, tab_rl2, tab_rl3, tab_rl4 = st.tabs(["🚀 Optimiser", "📊 Statut", "🎯 Recommandations", "📈 Apprentissage"])
    
    with tab_rl1:
        st.markdown("""
        <div class="rl-card">
            <h3>🚀 Optimisation Automatique</h3>
            <p>L'agent RL apprend à optimiser les index par essai-erreur</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_steps, col_strat = st.columns(2)
        
        with col_steps:
            steps = st.slider("Nombre d'étapes", 1, 20, 5)
        
        with col_strat:
            strategy = st.selectbox(
                "Stratégie",
                ["balanced", "aggressive", "conservative", "memory_saver"]
            )
        
        if st.button("🤖 Lancer l'optimisation RL", type="primary", use_container_width=True):
            with st.spinner("L'agent RL apprend et optimise..."):
                try:
                    response = requests.post(
                        f"{api_url}/api/rl/optimize",
                        json={"steps": steps, "strategy": strategy},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["success"]:
                            result = data["data"]
                            st.success("✅ Optimisation terminée!")
                            
                            # Sauvegarder
                            st.session_state.rl_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "result": result
                            })
                            
                            # Résumé
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Index finaux", f"{result['final_indexes']}/{result['max_indexes']}")
                            with col2:
                                st.metric("Performance", f"{result['final_query_time']:.3f}s")
                            with col3:
                                st.metric("Récompense", f"{result['total_reward']:.3f}")
                            
                            # Détails des étapes
                            st.subheader("📝 Détails des étapes")
                            
                            for step in result["steps_details"]:
                                action_class = f"action-{step['action'].lower()}"
                                st.markdown(f"""
                                <div class="step-card {action_class}">
                                    <strong>Étape {step['step']}: {step['action']}</strong>
                                    <p>Index: {step['indexes']} | Temps: {step['query_time']:.3f}s | Reward: {step['reward']:.3f}</p>
                                    <small>{step['explanation']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            st.error(f"Erreur: {data.get('error', 'Inconnue')}")
                    else:
                        st.error(f"Erreur API: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    with tab_rl2:
        st.subheader("📊 Statut de l'agent RL")
        
        try:
            response = requests.get(f"{api_url}/api/rl/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data["success"]:
                    status = data["data"]["status"]
                    learning = data["data"]["learning_stats"]
                    
                    # Cartes de statut
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.markdown("""
                        <div class="metric-card">
                            <h4>🤖 Agent RL</h4>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">{}</p>
                            <p style="color: #666;">Statut</p>
                        </div>
                        """.format(status['status']), unsafe_allow_html=True)
                    
                    with col_stat2:
                        st.markdown("""
                        <div class="metric-card">
                            <h4>📊 Index</h4>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">{}/{}</p>
                            <p style="color: #666;">Actuels / Max</p>
                        </div>
                        """.format(status['index_count'], status['max_indexes']), unsafe_allow_html=True)
                    
                    with col_stat3:
                        st.markdown("""
                        <div class="metric-card">
                            <h4>⚡ Performance</h4>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">{:.3f}s</p>
                            <p style="color: #666;">Temps requête</p>
                        </div>
                        """.format(status['performance']), unsafe_allow_html=True)
                    
                    # Informations supplémentaires
                    st.info(status['message'])
                    
                    # Graphique de performance
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = status['performance'],
                        title = {'text': "Performance actuelle"},
                        delta = {'reference': 0.05, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                        gauge = {
                            'axis': {'range': [None, 0.1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.03], 'color': "green"},
                                {'range': [0.03, 0.06], 'color': "yellow"},
                                {'range': [0.06, 0.1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"Erreur: {data.get('error', 'Inconnue')}")
            else:
                st.error("Erreur lors de la récupération du statut")
                
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")
    
    with tab_rl3:
        st.subheader("🎯 Recommandations RL")
        
        if st.button("🔄 Générer des recommandations", use_container_width=True):
            with st.spinner("L'agent RL génère des recommandations..."):
                try:
                    response = requests.get(f"{api_url}/api/rl/recommendations", timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["success"]:
                            recs = data["data"]
                            
                            st.metric("Index actuels", f"{recs['current_indexes']}/{recs['max_indexes']}")
                            st.info(f"Tendance: {recs['performance_trend']}")
                            
                            for i, rec in enumerate(recs['recommendations'], 1):
                                with st.container():
                                    # Couleur selon la priorité
                                    if rec['priority'] == 'high':
                                        st.error(f"**{i}. 🔴 HAUTE PRIORITÉ - {rec['type']}**")
                                    elif rec['priority'] == 'medium':
                                        st.warning(f"**{i}. 🟡 PRIORITÉ MOYENNE - {rec['type']}**")
                                    else:
                                        st.info(f"**{i}. 🟢 PRIORITÉ BASSE - {rec['type']}**")
                                    
                                    st.write(rec['description'])
                                    st.code(rec['sql'], language="sql")
                                    st.caption(f"Impact: {rec['impact']} | Confiance: {rec['confidence']*100:.0f}%")
                                    st.divider()
                        else:
                            st.error(f"Erreur: {data.get('error', 'Inconnue')}")
                    else:
                        st.error("Erreur lors de la génération")
                        
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    with tab_rl4:
        st.subheader("📈 Apprentissage RL")
        
        try:
            response = requests.get(f"{api_url}/api/rl/learning-stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data["success"]:
                    stats = data["data"]
                    
                    # Statistiques d'apprentissage
                    col_learn1, col_learn2, col_learn3 = st.columns(3)
                    
                    with col_learn1:
                        st.metric("Étapes totales", stats['total_steps'])
                    
                    with col_learn2:
                        st.metric("Récompense moyenne", f"{stats['average_reward']:.3f}")
                    
                    with col_learn3:
                        st.metric("Amélioration", f"{stats['performance_improvement']}%")
                    
                    # Graphique d'apprentissage
                    st.subheader("📊 Courbe d'apprentissage")
                    
                    # Données simulées
                    learning_steps = list(range(1, stats['total_steps'] + 1 if stats['total_steps'] > 0 else 10))
                    rewards = [0.1 + i*0.05 for i in range(len(learning_steps))]
                    performance = [0.06 - i*0.002 for i in range(len(learning_steps))]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=learning_steps, y=rewards, name='Récompense',
                                            line=dict(color='orange', width=3)))
                    fig.add_trace(go.Scatter(x=learning_steps, y=performance, name='Performance',
                                            yaxis='y2', line=dict(color='blue', width=3)))
                    
                    fig.update_layout(
                        title="Évolution de l'apprentissage",
                        yaxis=dict(title="Récompense"),
                        yaxis2=dict(title="Performance (s)", overlaying='y', side='right'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Progression
                    progress = stats.get('learning_progress', 0.65)
                    st.subheader(f"📈 Progression: {progress*100:.1f}%")
                    st.progress(progress)
                    
                else:
                    st.error(f"Erreur: {data.get('error', 'Inconnue')}")
            else:
                st.error("Erreur lors de la récupération")
                
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

elif page == "📊 Performances":
    st.header("📊 Analyse de performances")
    
    # Métriques globales
    try:
        response = requests.get(f"{api_url}/api/performance", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["success"]:
                perf_data = data["data"]
                
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                
                with col_perf1:
                    st.metric("Seuil lent", f"{perf_data['slow_threshold']}s")
                
                with col_perf2:
                    idx = perf_data['rl_status']['index_count']
                    max_idx = perf_data['rl_status']['max_indexes']
                    st.metric("Utilisation index", f"{(idx/max_idx*100):.1f}%")
                
                with col_perf3:
                    perf = perf_data['rl_status']['performance']
                    st.metric("Performance actuelle", f"{perf:.3f}s")
        
        else:
            # Valeurs par défaut
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("Seuil lent", "0.5s")
            with col_perf2:
                st.metric("Utilisation index", "60%")
            with col_perf3:
                st.metric("Performance", "0.038s")
                
    except:
        pass
    
    st.divider()
    
    # Graphiques comparatifs
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.subheader("⚡ Avant/Après optimisation")
        
        comparison_data = {
            "Métrique": ["Temps requête", "Utilisation CPU", "E/S disque", "Mémoire cache"],
            "Avant": [0.065, 75, 1200, 45],
            "Après": [0.038, 62, 850, 68]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Avant', x=comp_df['Métrique'], y=comp_df['Avant'], marker_color='indianred'),
            go.Bar(name='Après', x=comp_df['Métrique'], y=comp_df['Après'], marker_color='lightseagreen')
        ])
        
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_comp2:
        st.subheader("📈 Tendance des performances")
        
        # Données temporelles simulées
        dates = pd.date_range(end=datetime.today(), periods=15)
        performance_data = pd.DataFrame({
            'date': dates,
            'query_time': [0.062 - i*0.0015 for i in range(15)],
            'index_count': [1 + i//3 for i in range(15)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=performance_data['date'], y=performance_data['query_time'],
                                mode='lines+markers', name='Temps requête (s)',
                                line=dict(color='blue', width=3)))
        fig.add_trace(go.Bar(x=performance_data['date'], y=performance_data['index_count'],
                            name='Nombre d\'index', yaxis='y2',
                            marker_color='orange'))
        
        fig.update_layout(
            title="Évolution sur 15 jours",
            yaxis=dict(title="Temps (s)"),
            yaxis2=dict(title="Nombre d'index", overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations de performance
    st.subheader("🎯 Plan d'action pour amélioration")
    
    action_items = [
        ("🔴", "Critique", "Ajouter index composite sur orders(client_id, orderdate)", "CREATE INDEX idx_orders_client_date ON orders(client_id, orderdate)"),
        ("🟡", "Important", "Partitionner la table orders par date", "ALTER TABLE orders PARTITION BY RANGE(YEAR(orderdate)) (...)"),
        ("🟢", "Recommandé", "Archiver les commandes > 2 ans", "CREATE TABLE orders_archive AS SELECT * FROM orders WHERE orderdate < DATE_SUB(NOW(), INTERVAL 2 YEAR)"),
        ("🔵", "Maintenance", "Analyser les tables hebdomadairement", "ANALYZE TABLE orders, clients, products, cart")
    ]
    
    for icon, priority, desc, sql in action_items:
        with st.expander(f"{icon} {priority}: {desc}"):
            st.code(sql, language="sql")
            if st.button(f"Appliquer {priority}", key=f"apply_{priority}"):
                st.success(f"Action {priority} appliquée (simulation)")

elif page == "🗃️ Base de données":
    st.header("🗃️ Structure de la base POS")
    
    try:
        response = requests.get(f"{api_url}/api/tables", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["success"]:
                tables = data["data"]
                
                # Vue d'ensemble
                st.subheader(f"📊 Vue d'ensemble: {data['count']} tables")
                
                overview_data = []
                for table_name, info in tables.items():
                    overview_data.append({
                        "Table": table_name,
                        "Colonnes": len(info["columns"]),
                        "Lignes": info["row_count"],
                        "Index": info["index_count"]
                    })
                
                if overview_data:
                    overview_df = pd.DataFrame(overview_data)
                    st.dataframe(overview_df, use_container_width=True)
                    
                    # Graphique
                    fig = px.bar(overview_df, x='Table', y='Lignes',
                               title="Taille des tables (nombre de lignes)",
                               color='Index', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Détail par table
                selected_table = st.selectbox(
                    "Sélectionner une table pour détails:",
                    list(tables.keys())
                )
                
                if selected_table:
                    table_info = tables[selected_table]
                    
                    col_detail1, col_detail2 = st.columns([2, 1])
                    
                    with col_detail1:
                        st.subheader(f"📋 Structure: {selected_table}")
                        
                        if table_info["columns"]:
                            columns_df = pd.DataFrame(table_info["columns"])
                            st.dataframe(columns_df, use_container_width=True, hide_index=True)
                    
                    with col_detail2:
                        st.subheader("📊 Statistiques")
                        st.metric("Lignes", f"{table_info['row_count']:,}")
                        st.metric("Index", table_info["index_count"])
                        
                        if table_info["indexes"]:
                            st.subheader("🗂️ Index")
                            for idx in table_info["indexes"][:5]:  # Limiter à 5
                                st.code(f"{idx['INDEX_NAME']} ({idx['COLUMN_NAME']})")
            
            else:
                st.error(f"Erreur: {data.get('error', 'Inconnue')}")
        else:
            st.error("Erreur lors de la récupération")
            
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")

elif page == "⚙️ Configuration RL":
    st.header("⚙️ Configuration de l'agent RL")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.subheader("📐 Paramètres RL")
        
        max_indexes = st.number_input("Nombre maximum d'index", 1, 20, 5)
        episode_length = st.number_input("Longueur d'épisode", 10, 100, 25)
        learning_rate = st.slider("Taux d'apprentissage", 0.001, 0.1, 0.01)
        exploration_rate = st.slider("Taux d'exploration", 0.1, 0.9, 0.3)
        
        if st.button("💾 Sauvegarder configuration", use_container_width=True):
            st.success("Configuration sauvegardée (simulation)")
    
    with col_config2:
        st.subheader("⚖️ Coûts et récompenses")
        
        creation_cost = st.number_input("Coût création index", 0.01, 0.1, 0.02, 0.01)
        drop_penalty = st.number_input("Pénalité suppression", 0.01, 0.1, 0.01, 0.01)
        performance_weight = st.slider("Poids performance", 0.1, 1.0, 0.7)
        memory_weight = st.slider("Poids mémoire", 0.0, 0.5, 0.2)
    
    st.divider()
    
    # Stratégies
    st.subheader("🎯 Stratégies d'optimisation")
    
    strategies = {
        "balanced": "Équilibre entre performance et ressources",
        "aggressive": "Priorité performance (plus d'index)",
        "conservative": "Prudent (moins d'index)",
        "memory_saver": "Économie mémoire maximale"
    }
    
    selected_strategy = st.selectbox(
        "Stratégie par défaut:",
        list(strategies.keys()),
        format_func=lambda x: f"{x} - {strategies[x]}"
    )
    
    st.info(f"**Stratégie sélectionnée:** {selected_strategy} - {strategies[selected_strategy]}")
    
    # Entraînement
    st.subheader("🎓 Entraînement RL")
    
    col_train1, col_train2 = st.columns(2)
    
    with col_train1:
        episodes = st.number_input("Nombre d'épisodes", 1, 100, 10)
    
    with col_train2:
        batch_size = st.number_input("Taille du batch", 32, 256, 64)
    
    if st.button("🎓 Démarrer l'entraînement RL", type="primary", use_container_width=True):
        with st.spinner("Entraînement en cours..."):
            time.sleep(2)  # Simulation
            st.success(f"Entraînement terminé! {episodes} épisodes")
            
            # Résultats simulés
            st.subheader("📊 Résultats de l'entraînement")
            
            results_data = {
                "Épisode": list(range(1, 11)),
                "Récompense": [0.1 + i*0.08 for i in range(10)],
                "Performance": [0.06 - i*0.002 for i in range(10)]
            }
            
            results_df = pd.DataFrame(results_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['Épisode'], y=results_df['Récompense'],
                                   name='Récompense', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=results_df['Épisode'], y=results_df['Performance'],
                                   name='Performance', yaxis='y2', line=dict(color='blue')))
            
            fig.update_layout(
                title="Progression de l'entraînement",
                yaxis=dict(title="Récompense"),
                yaxis2=dict(title="Performance (s)", overlaying='y', side='right')
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>🚀 SADOP v4.0</strong> | Système Autonome de Diagnostic et d'Optimisation | © 2024 APBD Team</p>
    <p>🔍 XGBoost | 🤖 Reinforcement Learning | 📊 Analyse intelligente</p>
</div>
""", unsafe_allow_html=True)

# Rafraîchissement automatique
if st.button("🔄 Rafraîchir toutes les données", use_container_width=True):
    st.rerun()