import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="SADOP - Interface Intelligente",
    page_icon="⚡",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 12px;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        padding: 20px;
        background-color: #fff3cd;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("⚡ SADOP - Système Autonome de Diagnostic et d'Optimisation")
st.markdown("**Interface d'administration intelligente pour bases de données MySQL**")

# Initialiser l'historique des conversations
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4299/4299756.png", width=100)
    st.header("📊 Configuration")
    
    api_url = st.text_input(
        "URL de l'API Backend",
        value="http://localhost:8000",
        help="URL de votre API FastAPI"
    )
    
    st.divider()
    
    # Exemples de questions
    st.header("💡 Exemples de questions")
    examples = [
        "Analyser cette requête: SELECT * FROM orders WHERE customer_id = 123",
        "Comment optimiser la table 'transactions'?",
        "Quelles sont les requêtes les plus lentes aujourd'hui?",
        "Faut-il ajouter un index sur customer_id?",
        "Explique-moi le plan d'exécution de SELECT * FROM products"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:10]}"):
            st.session_state.user_question = example

# Deux colonnes principales
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation avec SADOP")
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ⚠️ IMPORTANT: st.chat_input() doit être EN DEHORS des colonnes
# Placez-le APRÈS la fermeture de with col1:

# Zone de saisie GLOBALE (pas dans une colonne)
user_question = st.chat_input("Posez votre question en langage naturel...")

if user_question:
    # Afficher la question de l'utilisateur
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Sauvegarder dans l'historique
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("🧠 L'agent SADOP analyse votre question..."):
            try:
                # Appel à l'API
                response = requests.post(
                    f"{api_url}/chat",
                    json={"question": user_question, "user_id": "streamlit_user"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["response"]
                    
                    # Afficher la réponse
                    st.markdown(answer)
                    
                    # Sauvegarder
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    
                    # Boutons d'action (optionnel)
                    st.markdown("---")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📋 Copier", key=f"copy_{len(st.session_state.messages)}"):
                            st.code(answer, language="markdown")
                    with col_b:
                        if st.button("🔄 Nouveau", key=f"new_{len(st.session_state.messages)}"):
                            st.session_state.messages = []
                            st.rerun()
                else:
                    st.error(f"❌ Erreur API: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Impossible de se connecter à l'API. Vérifiez que le backend est démarré.")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# Le reste du code (col2, sidebar, footer) reste APRÈS
with col2:
    st.header("📈 Tableau de bord")
    
    # Métriques
    st.metric("Conversations", len(st.session_state.conversation_history))
    st.metric("Requêtes analysées", "12")  # À remplacer par vos données
    
    st.divider()
    
    # Actions rapides
    st.subheader("⚡ Actions rapides")
    
    if st.button("🔍 Diagnostic complet", key="full_diag"):
        st.info("Cette fonctionnalité analyserait toutes les requêtes récentes")
    
    if st.button("📊 Voir les index", key="view_indexes"):
        st.info("Afficherait tous les indexes de la base de données")
    
    if st.button("🔄 Rafraîchir les métriques", key="refresh"):
        st.rerun()
    
    st.divider()
    
    # Historique des conversations
    st.subheader("📜 Historique récent")
    if st.session_state.conversation_history:
        for i, conv in enumerate(st.session_state.conversation_history[-3:]):
            st.caption(f"**Q:** {conv['question'][:50]}...")
    else:
        st.caption("Aucune conversation encore")

# Pied de page
st.markdown("---")
st.caption("SADOP v1.0 | Système Autonome de Diagnostic et d'Optimisation de Performance")

if __name__ == "__main__":
    # Vérifier la connexion API
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            st.sidebar.success("✅ API connectée")
        else:
            st.sidebar.warning("⚠️ API non disponible")
    except:
        st.sidebar.error("🔌 API non connectée")