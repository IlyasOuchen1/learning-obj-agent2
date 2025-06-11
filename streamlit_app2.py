import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
import json
import uuid
from enhanced_agent import EnhancedLearningObjectiveAgent

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "./outputs")

def initialize_session():
    """Initialise la session et l'agent automatiquement"""
    # Initialise toutes les variables de session
    # Crée automatiquement l'agent si la clé API est disponible


# Importer la configuration
try:
    from config import OPENAI_API_KEY, LLM_MODEL, OUTPUT_DIRECTORY
except ImportError:
    # Fallback si config.py n'est pas disponible
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
    OUTPUT_DIRECTORY = os.environ.get("OUTPUT_DIRECTORY", "./outputs")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Configuration de la page
st.set_page_config(
    page_title="🎯 Analyseur d'Objectifs d'Apprentissage",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une meilleure structure et présentation
st.markdown("""
<style>
    /* Fond sombre pour correspondre à l'image de référence */
    .stApp {
        background-color: #111827 !important;
        color: #f9fafb !important;
    }
    
    /* Styles généraux */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f9fafb !important;
    }
    
    /* Adaptation des couleurs pour le thème sombre */
    p, li, label, .stMarkdown {
        color: #e5e7eb !important;
    }
    
    /* Correction pour tous les éléments avec fond blanc */
    div[data-testid="stMetricValue"] > div {
        background-color: #1e293b !important;
        color: #f9fafb !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Correction des métriques */
    div[data-testid="metric-container"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    }
    
    div[data-testid="metric-container"] > div {
        background-color: transparent !important;
    }
    
    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 0.9rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #f9fafb !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Correction des colonnes et containers */
    div[data-testid="column"] > div {
        background-color: transparent !important;
    }
    
    /* Correction des éléments de contenu */
    .element-container {
        background-color: transparent !important;
    }
    
    /* Correction des blocks de contenu avec fond blanc */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }
    
    /* Correction spécifique pour les cartes blanches */
    div[style*="background-color: white"], 
    div[style*="background-color: #ffffff"],
    div[style*="background-color: #fff"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f9fafb !important;
    }
    
    /* En-tête de l'application */
    .app-header {
        padding: 1.5rem 0;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f9fafb !important;
        margin-bottom: 0.5rem;
    }
    
    .app-description {
        font-size: 1.1rem;
        color: #d1d5db !important;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Carte de formulaire */
    .form-card {
        background-color: #1e293b !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .form-card:hover {
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        border-color: #475569;
    }
    
    /* Amélioration des champs de saisie pour thème sombre */
    .stTextArea textarea {
        background-color: #273549 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        color: #f9fafb !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4b6fff !important;
        box-shadow: 0 0 0 3px rgba(75, 111, 255, 0.3) !important;
        background-color: #1e293b !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    
    /* Étiquette du champ avec style amélioré */
    .stTextArea label, .stTextInput label, .stSelectbox label {
        font-weight: 500 !important;
        color: #f9fafb !important;
        margin-bottom: 8px !important;
    }
    
    /* Amélioration des autres widgets de saisie */
    .stTextInput input {
        background-color: #273549 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 6px !important;
        color: #f9fafb !important;
    }
    
    .stTextInput input:focus {
        border-color: #4b6fff !important;
        box-shadow: 0 0 0 3px rgba(75, 111, 255, 0.3) !important;
    }
    
    .stSelectbox select {
        background-color: #273549 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 6px !important;
        color: #f9fafb !important;
    }
    
    .stFileUploader {
        background-color: #273549 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 6px !important;
        padding: 5px !important;
    }
    
    .stFileUploader:hover {
        border-color: #4b6fff !important;
    }
    
    /* Styles pour les onglets de résultats */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding: 10px 16px;
        background-color: #374151 !important;
        color: #d1d5db !important;
        font-weight: 500;
        border: 1px solid #4b5563 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4b6fff !important;
        color: white !important;
        border-color: #4b6fff !important;
    }
    
    /* Badges pour les niveaux de Bloom */
    .bloom-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .se-souvenir { background-color: #7f1d1d; color: #fecaca; }
    .comprendre { background-color: #164e63; color: #bae6fd; }
    .appliquer { background-color: #14532d; color: #bbf7d0; }
    .analyser { background-color: #78350f; color: #fef3c7; }
    .evaluer { background-color: #581c87; color: #e9d5ff; }
    .creer { background-color: #831843; color: #f9d7e9; }
    
    /* Cartes de métrique */
    .metric-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        text-align: center;
        height: 100%;
        color: #f9fafb !important;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #9ca3af !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f9fafb !important;
    }
    
    .metric-value.level {
        font-size: 1.75rem;
    }
    
    /* Cartes de document */
    .document-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #f9fafb !important;
    }
    
    /* Messages de statut */
    .success-message {
        background-color: #065f46 !important;
        color: #a7f3d0 !important;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .warning-message {
        background-color: #92400e !important;
        color: #fde68a !important;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    /* Correction pour la sidebar */
    .css-1d391kg {
        background-color: #1e293b !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #f9fafb !important;
    }
    
    /* Correction pour les sliders */
    .stSlider > div > div {
        background-color: #273549 !important;
    }
    
    /* Correction pour les checkboxes */
    .stCheckbox > label {
        color: #f9fafb !important;
    }
    
    /* Correction pour les boutons de download */
    .stDownloadButton > button {
        background-color: #4b6fff !important;
        color: white !important;
        border: none !important;
    }
    
    /* Correction pour les expanders */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #f9fafb !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    
    /* Force tous les divs avec background blanc à être sombres */
    div[style*="background: white"],
    div[style*="background: #ffffff"],
    div[style*="background: #fff"],
    div[style*="background-color: white"],
    div[style*="background-color: #ffffff"],
    div[style*="background-color: #fff"] {
        background-color: #1e293b !important;
        color: #f9fafb !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #1e293b !important;
    }
    
    /* Caption styling */
    .caption {
        color: #9ca3af !important;
        font-size: 0.875rem !important;
    }
    
    /* Styles pour les résultats */
    .result-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: #f9fafb !important;
    }
    
    .objective-card {
        background-color: #1e293b !important;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []

# Fonctions utilitaires pour l'extraction de données
def extract_difficulty_level(difficulty_text, objective):
    """Extrait le niveau de difficulté d'un objectif"""
    pattern = re.compile(rf"{re.escape(objective)}.*?niveau de difficulté.*?(\d)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(difficulty_text)
    if match:
        return int(match.group(1))
    return 3  # Niveau moyen par défaut

def extract_bloom_level(classification_text, objective):
    """Extrait le niveau de Bloom d'un objectif"""
    pattern = re.compile(rf"{re.escape(objective)}.*?Niveau de Bloom: ([a-zéè ]+)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(classification_text)
    if match:
        level = match.group(1).strip().lower()
        return level
    return "non classifié"

def create_bloom_distribution_chart(objectives_with_levels):
    """Crée le graphique de distribution des niveaux de Bloom"""
    # Comptage des niveaux
    level_counts = {}
    for level in objectives_with_levels.values():
        level = level.lower().strip()
        if level in level_counts:
            level_counts[level] += 1
        else:
            level_counts[level] = 1
    
    # Création du DataFrame pour le graphique
    df = pd.DataFrame({
        'Niveau': list(level_counts.keys()),
        'Nombre d\'objectifs': list(level_counts.values())
    })
    
    # Ordre des niveaux de Bloom
    bloom_order = ["se souvenir", "comprendre", "appliquer", "analyser", "évaluer", "créer"]
    
    # Filtrer et trier le DataFrame selon l'ordre de Bloom
    df = df[df['Niveau'].isin(bloom_order)]
    df['Niveau'] = pd.Categorical(df['Niveau'], categories=bloom_order, ordered=True)
    df = df.sort_values('Niveau')
    
    # Création du graphique avec couleurs adaptées au thème sombre
    colors = {
        "se souvenir": "#ef4444", 
        "comprendre": "#3b82f6", 
        "appliquer": "#10b981", 
        "analyser": "#f59e0b", 
        "évaluer": "#8b5cf6", 
        "créer": "#ec4899"
    }
    
    fig = px.bar(
        df, 
        x='Niveau', 
        y='Nombre d\'objectifs',
        title='Distribution des niveaux de la taxonomie de Bloom',
        color='Niveau',
        color_discrete_map=colors,
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Niveau de Bloom",
        yaxis_title="Nombre d'objectifs",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#f9fafb',
        yaxis=dict(gridcolor='#374151'),
        xaxis=dict(gridcolor='#374151'),
        font=dict(family="Segoe UI, Arial", size=12, color='#f9fafb'),
        title=dict(font=dict(family="Segoe UI, Arial", size=16, color='#f9fafb')),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def create_difficulty_chart(objectives_with_difficulty):
    """Crée le graphique de distribution des difficultés"""
    # Comptage des niveaux de difficulté
    difficulty_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for level in objectives_with_difficulty.values():
        if level in difficulty_counts:
            difficulty_counts[level] += 1
    
    # Création du DataFrame pour le graphique
    df = pd.DataFrame({
        'Niveau de difficulté': list(difficulty_counts.keys()),
        'Nombre d\'objectifs': list(difficulty_counts.values())
    })
    
    # Définition des couleurs pour chaque niveau
    colors = {
        1: "#3b82f6",  # Bleu
        2: "#10b981",  # Vert
        3: "#f59e0b",  # Jaune/Orange
        4: "#f97316",  # Orange
        5: "#ef4444"   # Rouge
    }
    
    # Création du graphique
    fig = px.bar(
        df, 
        x='Niveau de difficulté', 
        y='Nombre d\'objectifs',
        title='Distribution des niveaux de difficulté',
        color='Niveau de difficulté',
        color_discrete_map={str(k): v for k, v in colors.items()},
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Niveau de difficulté",
        yaxis_title="Nombre d'objectifs",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#f9fafb',
        yaxis=dict(gridcolor='#374151'),
        xaxis=dict(gridcolor='#374151'),
        font=dict(family="Segoe UI, Arial", size=12, color='#f9fafb'),
        title=dict(font=dict(family="Segoe UI, Arial", size=16, color='#f9fafb')),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def bloom_badge(level):
    """Crée un badge coloré pour un niveau de Bloom"""
    level_class = level.lower().replace(" ", "-").replace("é", "e").replace("è", "e")
    return f'<span class="bloom-badge {level_class}">{level.capitalize()}</span>'

def display_app_header():
    """Affiche l'en-tête de l'application"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🎯 Analyseur d'Objectifs d'Apprentissage</h1>
        <p class="app-description">
            Analysez et améliorez vos objectifs pédagogiques avec l'IA et la taxonomie de Bloom. 
            Enrichissez l'analyse en téléchargeant vos documents pédagogiques.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_document_manager():
    """Affiche la section de gestion des documents avec extraction d'objectifs"""
    st.header("📚 Gestion des Documents Pédagogiques")
    
    # Description améliorée
    st.markdown("""
    Téléchargez vos documents pédagogiques (syllabus, plans de cours, etc.) pour enrichir l'analyse. 
    L'IA va automatiquement **identifier les objectifs d'apprentissage** déjà présents dans vos documents !
    """)
    
    # Upload de fichiers
    uploaded_files = st.file_uploader(
        "Sélectionnez vos fichiers",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Formats supportés: PDF, TXT, DOCX. L'IA extraira automatiquement les objectifs existants."
    )
    
    # Actions sur les fichiers
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        process_button = st.button(
            "📥 Traiter les fichiers", 
            disabled=not uploaded_files or not st.session_state.agent,
            use_container_width=True
        )
    
    with col2:
        clear_button = st.button(
            "🗑️ Effacer la session",
            type="secondary",
            use_container_width=True
        )
    
    with col3:
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} fichier(s) sélectionné(s)")
        elif st.session_state.processed_documents:
            st.success(f"✅ {len(st.session_state.processed_documents)} document(s) traité(s)")
            
            # Afficher le nombre d'objectifs extraits si disponible
            if hasattr(st.session_state, 'extracted_objectives_count'):
                st.metric(
                    "🎯 Objectifs extraits", 
                    st.session_state.extracted_objectives_count,
                    help="Objectifs d'apprentissage trouvés dans les documents"
                )
    
    # Traitement des fichiers avec extraction d'objectifs
    if process_button and uploaded_files and st.session_state.agent:
        with st.spinner("⏳ Traitement et extraction d'objectifs en cours..."):
            try:
                # Étapes de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Étape 1: Traitement des fichiers
                status_text.text("📄 Traitement des fichiers...")
                progress_bar.progress(20)
                
                result = st.session_state.agent.process_uploaded_files(
                    uploaded_files, 
                    st.session_state.session_id
                )
                
                # Étape 2: Extraction des objectifs
                status_text.text("🎯 Extraction des objectifs d'apprentissage...")
                progress_bar.progress(60)
                
                # Récupérer les objectifs extraits avec gestion d'erreur
                try:
                    extracted_objectives = st.session_state.agent.get_extracted_document_objectives()
                except AttributeError:
                    extracted_objectives = []
                    print("Méthode get_extracted_document_objectives() non disponible")
                
                # Étape 3: Finalisation
                status_text.text("✅ Finalisation...")
                progress_bar.progress(100)
                
                # Nettoyage de l'interface
                progress_bar.empty()
                status_text.empty()
                
                if "✅" in result:
                    st.success(result)
                    st.session_state.processed_documents = st.session_state.agent.get_processed_documents_summary()
                    st.session_state.extracted_objectives_count = len(extracted_objectives)
                    
                    # Affichage spécial si des objectifs ont été trouvés
                    if extracted_objectives:
                        st.balloons()  # Animation de célébration
                        st.success(f"🎉 Excellent ! {len(extracted_objectives)} objectif(s) d'apprentissage identifié(s) automatiquement !")
                        
                        # Aperçu des objectifs trouvés avec gestion d'erreur
                        with st.expander(f"👀 Aperçu des {len(extracted_objectives)} objectifs trouvés", expanded=True):
                            for i, obj in enumerate(extracted_objectives[:5], 1):  # Afficher les 5 premiers
                                # Gestion sécurisée des clés avec valeurs par défaut
                                objective_text = obj.get('objective', 'Objectif non spécifié')
                                obj_type = obj.get('type', 'non classifié')
                                source_doc = obj.get('document_source', obj.get('source_document', 'Document inconnu'))
                                
                                type_icon = "🎯" if obj_type == "explicite" else "💡"
                                
                                st.markdown(f"""
                                <div style="background-color: #1e293b; border-left: 4px solid #10b981; padding: 0.75rem; margin: 0.5rem 0; border-radius: 6px;">
                                    <strong>{type_icon} Objectif {i}:</strong> {objective_text}<br>
                                    <small style="color: #9ca3af;">📄 Source: {source_doc} | Type: {obj_type}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if len(extracted_objectives) > 5:
                                st.info(f"... et {len(extracted_objectives) - 5} autre(s) objectif(s). Consultez l'analyse complète après avoir lancé l'analyse.")
                    else:
                        st.info("📝 Aucun objectif d'apprentissage explicite trouvé dans les documents. Cela n'empêche pas l'enrichissement de l'analyse !")
                    
                    # Sauvegarder les informations de session
                    import json
                    session_data = {
                        "session_id": st.session_state.session_id,
                        "processed_documents": st.session_state.processed_documents,
                        "extracted_objectives_count": len(extracted_objectives),
                        "extracted_objectives": extracted_objectives
                    }
                    
                    try:
                        os.makedirs("./outputs", exist_ok=True)
                        with open(f"./outputs/session_{st.session_state.session_id[:8]}.json", "w", encoding="utf-8") as f:
                            json.dump(session_data, f, ensure_ascii=False, indent=2)
                    except Exception as save_error:
                        print(f"Erreur de sauvegarde: {save_error}")
                    
                    st.rerun()
                else:
                    st.error(result)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")
                st.info("💡 Vérifiez que vos documents contiennent du texte lisible et réessayez.")
    
    # Effacement de la session
    if clear_button:
        if st.session_state.agent:
            st.session_state.agent.clear_session()
        st.session_state.processed_documents = []
        st.session_state.session_id = str(uuid.uuid4())
        if hasattr(st.session_state, 'extracted_objectives_count'):
            delattr(st.session_state, 'extracted_objectives_count')
        st.success("🗑️ Session effacée avec succès")
        st.rerun()
    
    # Affichage détaillé des documents traités avec objectifs
    if st.session_state.processed_documents:
        st.subheader("📄 Documents Traités")
        
        # Récupérer les objectifs extraits pour affichage avec gestion d'erreur
        extracted_objectives = []
        try:
            if st.session_state.agent and hasattr(st.session_state.agent, 'extracted_document_objectives'):
                extracted_objectives = st.session_state.agent.extracted_document_objectives
            elif st.session_state.agent:
                # Essayer la méthode get_extracted_document_objectives
                extracted_objectives = st.session_state.agent.get_extracted_document_objectives()
        except (AttributeError, Exception) as e:
            print(f"Erreur lors de la récupération des objectifs: {e}")
            extracted_objectives = []
        
        # Organiser les objectifs par document avec gestion d'erreur
        objectives_by_doc = {}
        for obj in extracted_objectives:
            try:
                # Essayer différentes clés possibles pour le nom du document
                doc_name = obj.get('document_source') or obj.get('source_document') or obj.get('source') or 'Document inconnu'
                
                if doc_name not in objectives_by_doc:
                    objectives_by_doc[doc_name] = []
                objectives_by_doc[doc_name].append(obj)
            except Exception as e:
                print(f"Erreur lors de l'organisation des objectifs: {e}")
                continue
        
        # Affichage pour chaque document
        for i, doc in enumerate(st.session_state.processed_documents, 1):
            doc_name = doc['name']
            doc_objectives = objectives_by_doc.get(doc_name, [])
            
            # Titre avec indicateur d'objectifs
            if doc_objectives:
                title = f"📄 {doc_name} • {len(doc_objectives)} objectif(s) trouvé(s) 🎯"
                expanded = True  # Ouvrir automatiquement si des objectifs sont trouvés
            else:
                title = f"📄 {doc_name} • Aucun objectif identifié"
                expanded = False
            
            with st.expander(title, expanded=expanded):
                # Informations sur le document
                st.markdown(f"""
                <div class="document-card">
                    <strong>📁 Nom du fichier:</strong> {doc_name}<br>
                    <strong>📊 Statut:</strong> {'✅ Objectifs trouvés' if doc_objectives else '📝 Contenu analysé'}<br>
                    <strong>📝 Aperçu du contenu:</strong><br>
                    <em>{doc.get('text_preview', 'Aucun aperçu disponible')}</em>
                </div>
                """, unsafe_allow_html=True)
                
                # Affichage des objectifs si trouvés
                if doc_objectives:
                    st.markdown("**🎯 Objectifs d'apprentissage identifiés:**")
                    for j, obj in enumerate(doc_objectives, 1):
                        # Gestion sécurisée de toutes les clés
                        try:
                            objective_text = obj.get('objective', 'Objectif non spécifié')
                            obj_type = obj.get('type', 'non classifié')
                            source_text = obj.get('source_text', obj.get('source', 'Texte source non disponible'))
                            
                            type_color = "#10b981" if obj_type == "explicite" else "#f59e0b"
                            type_icon = "🎯" if obj_type == "explicite" else "💡"
                            
                            st.markdown(f"""
                            <div style="background-color: #273549; border-left: 3px solid {type_color}; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong>Objectif {j}:</strong>
                                    <span style="background-color: {type_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.75rem;">
                                        {type_icon} {obj_type.capitalize()}
                                    </span>
                                </div>
                                <div style="margin-bottom: 0.5rem; font-weight: 500;">
                                    "{objective_text}"
                                </div>
                                <details style="margin-top: 0.5rem;">
                                    <summary style="cursor: pointer; color: #9ca3af; font-size: 0.9rem;">📄 Voir l'extrait source</summary>
                                    <div style="background-color: #1e293b; padding: 0.5rem; margin-top: 0.5rem; border-radius: 4px; font-style: italic; font-size: 0.85rem;">
                                        "{source_text[:300]}{'...' if len(source_text) > 300 else ''}"
                                    </div>
                                </details>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as obj_error:
                            st.error(f"Erreur lors de l'affichage de l'objectif {j}: {obj_error}")
                            # Affichage minimal en cas d'erreur
                            st.markdown(f"""
                            <div style="background-color: #374151; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;">
                                <strong>⚠️ Objectif {j}:</strong> Erreur d'affichage<br>
                                <small>Données disponibles: {list(obj.keys())}</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("💡 Ce document ne contient pas d'objectifs d'apprentissage explicites, mais son contenu enrichira quand même l'analyse globale.")
    
    # Conseils pour de meilleurs résultats
    with st.expander("💡 Conseils pour optimiser l'extraction d'objectifs"):
        st.markdown("""
        **🎯 Types de documents recommandés:**
        - Syllabus de cours avec section "Objectifs"
        - Plans de cours détaillés
        - Documents contenant "Learning objectives" ou "Compétences visées"
        - Guides pédagogiques avec objectifs explicites
        
        **✅ Formulations d'objectifs bien détectées:**
        - "À la fin de ce cours, l'étudiant sera capable de..."
        - "Les objectifs de ce module sont..."
        - "By the end of this course, students will be able to..."
        - "Compétences visées : l'apprenant devra..."
        
        **📝 Si aucun objectif n'est trouvé:**
        - Le document ne contient peut-être pas d'objectifs explicites
        - Les objectifs sont formulés de manière non-standard
        - Le contenu reste utile pour enrichir l'analyse globale
        
        **🔄 L'IA analyse en continu:**
        - Recherche par mots-clés pédagogiques
        - Détection de patterns d'objectifs
        - Classification automatique explicite/implicite
        """)
    
    st.divider()

def display_input_form():
    """Affiche le formulaire de saisie des informations"""
    st.header("📝 Informations sur votre Cours")
    
    # Section 1: Sujet du cours (obligatoire)
    st.markdown("### 🎯 Sujet Principal *")
    course_subject = st.text_area(
        "Décrivez le sujet principal et le contenu général de votre cours",
        height=120,
        placeholder="Exemple: Ce cours porte sur les principes fondamentaux de l'intelligence artificielle, couvrant l'apprentissage automatique, les réseaux de neurones et les applications pratiques de l'IA dans différents domaines...",
        key="course_subject",
        help="Soyez aussi précis que possible sur le sujet, les concepts clés, et les compétences visées."
    )
    
    # Section 2: Public cible (optionnel)
    st.markdown("### 👥 Public Cible")
    target_audience = st.text_area(
        "Décrivez votre public cible (niveau, prérequis, contexte, etc.)",
        height=100,
        placeholder="Exemple: Étudiants de niveau licence en informatique ayant des connaissances de base en programmation Python et en mathématiques (algèbre linéaire et statistiques)...",
        key="target_audience",
        help="Le niveau et les connaissances préalables influenceront la difficulté et la formulation des objectifs."
    )
    
    # Section 3: Objectifs existants (optionnel)
    st.markdown("### 🎯 Objectifs d'Apprentissage Existants")
    learning_objectives = st.text_area(
        "Listez vos objectifs d'apprentissage actuels (un par ligne)",
        height=150,
        placeholder="Exemple:\n- Comprendre les principes fondamentaux de l'IA\n- Implémenter des algorithmes d'apprentissage automatique simples\n- Analyser les performances des modèles d'IA\n- Évaluer les implications éthiques de l'IA",
        key="learning_objectives",
        help="Si vous n'avez pas encore d'objectifs définis, laissez ce champ vide. L'IA les générera à partir des autres informations."
    )
    
    # Section 4: Contenu supplémentaire (optionnel)
    st.markdown("### 📄 Contenu Supplémentaire")
    source_text = st.text_area(
        "Collez ici tout contenu additionnel (plan de cours, descriptions de modules, etc.)",
        height=120,
        placeholder="Collez ici du contenu supplémentaire comme un plan de cours détaillé, des descriptions de modules, des compétences visées, etc.",
        key="source_text",
        help="Plus vous fournissez de contenu, plus l'analyse sera précise et personnalisée."
    )
    
    # Boutons d'action
    st.markdown("### 🚀 Actions")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        submit_button = st.button(
            "🔍 Analyser",
            type="primary",
            use_container_width=True,
            help="Lance l'analyse complète des objectifs d'apprentissage"
        )
    
    with col2:
        clear_button = st.button(
            "🧹 Effacer le formulaire",
            use_container_width=True,
            help="Efface tous les champs du formulaire"
        )
    
    with col3:
        st.markdown("""
        <div class="caption">
            💡 L'analyse peut prendre 1-2 minutes selon la quantité de contenu fournie.
            Les documents uploadés enrichiront automatiquement l'analyse.
        </div>
        """, unsafe_allow_html=True)
    
    return {
        "course_subject": course_subject,
        "target_audience": target_audience,
        "learning_objectives": learning_objectives,
        "source_text": source_text,
        "submit": submit_button,
        "clear": clear_button
    }

def display_metrics(num_objectives, highest_level, avg_difficulty):
    """Affiche les métriques principales"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Objectifs identifiés</div>
            <div class="metric-value">{num_objectives}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        highest_level_cap = highest_level.capitalize() if highest_level != "Non disponible" else highest_level
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Niveau Bloom dominant</div>
            <div class="metric-value level">{highest_level_cap}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Difficulté moyenne</div>
            <div class="metric-value">{avg_difficulty:.1f}<span style="font-size: 1.25rem; color: #9ca3af;">/5</span></div>
        </div>
        """, unsafe_allow_html=True)

def display_results():
    """Affiche les résultats de l'analyse avec tous les éléments"""
    results = st.session_state.results
    
    # Bouton de retour
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Retour au formulaire", type="primary"):
            st.session_state.analysis_done = False
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Résultats de l'Analyse")
    
    st.divider()
    
    # Vérification des erreurs
    if "error" in results and results["error"]:
        st.error(f"❌ Une erreur s'est produite lors de l'analyse: {results['error']}")
        st.markdown("""
        **Suggestions pour résoudre le problème:**
        - Vérifiez votre clé API OpenAI
        - Réduisez la quantité de contenu si elle est très importante
        - Réessayez dans quelques minutes
        """)
        return
    
    # Extraction des données des résultats
    all_objectives = results["objectives"]
    content_objectives = results.get("content_objectives", [])
    document_objectives = results.get("document_objectives", [])
    content_analysis = results["content_analysis"]["analysis"]
    classification = results["classification"]["classification"] 
    formatted_objectives = results["formatted_objectives"]["formatted_objectives"]
    difficulty_evaluation = results["difficulty_evaluation"]["difficulty_evaluation"]
    recommendations = results["recommendations"]["recommendations"]
    feedback = results["feedback"]["feedback"]
    stats = results.get("stats", {})
    
    # Vérification que nous avons des objectifs
    if not all_objectives:
        st.warning("⚠️ Aucun objectif d'apprentissage n'a pu être extrait du contenu fourni.")
        st.markdown("""
        **Suggestions:**
        - Ajoutez plus de détails sur le contenu de votre cours
        - Incluez des objectifs explicites si vous en avez
        - Téléchargez des documents pédagogiques pour enrichir l'analyse
        """)
        return
    
    # Affichage des statistiques détaillées
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Objectifs", 
            stats.get("total_objectives", len(all_objectives)),
            help="Nombre total d'objectifs identifiés"
        )
    
    with col2:
        st.metric(
            "📝 Du Contenu", 
            stats.get("content_objectives_count", len(content_objectives)),
            help="Objectifs extraits du contenu saisi"
        )
    
    with col3:
        st.metric(
            "📄 Des Documents", 
            stats.get("document_objectives_count", len(document_objectives)),
            help="Objectifs trouvés dans les documents uploadés"
        )
    
    with col4:
        st.metric(
            "📚 Documents Analysés", 
            stats.get("processed_documents", len(st.session_state.processed_documents)),
            help="Nombre de documents pédagogiques traités"
        )
    
    # Bouton de téléchargement JSON
    st.download_button(
        label="📥 Télécharger les résultats (JSON)",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name=f"analyse_objectifs_{st.session_state.session_id[:8]}.json",
        mime="application/json",
        help="Téléchargez tous les résultats de l'analyse au format JSON"
    )
    
    st.divider()
    
    # Vérification que nous avons des objectifs
    if not all_objectives:
        st.warning("⚠️ Aucun objectif d'apprentissage n'a pu être extrait du contenu fourni.")
        st.markdown("""
        **Suggestions:**
        - Ajoutez plus de détails sur le contenu de votre cours
        - Incluez des objectifs explicites si vous en avez
        - Téléchargez des documents pédagogiques pour enrichir l'analyse
        """)
        return
    
    # Affichage des statistiques détaillées
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Objectifs", 
            stats.get("total_objectives", len(all_objectives)),
            help="Nombre total d'objectifs identifiés"
        )
    
    with col2:
        st.metric(
            "📝 Du Contenu", 
            stats.get("content_objectives_count", len(content_objectives)),
            help="Objectifs extraits du contenu saisi"
        )
    
    with col3:
        st.metric(
            "📄 Des Documents", 
            stats.get("document_objectives_count", len(document_objectives)),
            help="Objectifs trouvés dans les documents uploadés"
        )
    
    with col4:
        st.metric(
            "📚 Documents Analysés", 
            stats.get("processed_documents", len(st.session_state.processed_documents)),
            help="Nombre de documents pédagogiques traités"
        )
    
    # Message informatif sur la source des objectifs
    if document_objectives:
        st.info(f"🎯 Analyse enrichie ! {len(document_objectives)} objectif(s) supplémentaire(s) trouvé(s) dans vos documents pédagogiques.")
    
    # Extraction des niveaux pour les graphiques et métriques
    objectives_with_levels = {}
    objectives_with_difficulty = {}
    
    for obj in all_objectives:
        bloom_level = extract_bloom_level(classification, obj)
        objectives_with_levels[obj] = bloom_level
        
        difficulty_level = extract_difficulty_level(difficulty_evaluation, obj)
        objectives_with_difficulty[obj] = difficulty_level
    
    # Calculs pour les métriques
    bloom_levels_count = {}
    for level in objectives_with_levels.values():
        bloom_levels_count[level] = bloom_levels_count.get(level, 0) + 1
    
    highest_level = max(bloom_levels_count.items(), key=lambda x: x[1])[0] if bloom_levels_count else "Non disponible"
    avg_difficulty = sum(objectives_with_difficulty.values()) / len(objectives_with_difficulty) if objectives_with_difficulty else 0
    
    # Affichage des métriques principales
    display_metrics(len(all_objectives), highest_level, avg_difficulty)
    
    st.divider()
    
    # Onglets des résultats avec tous les éléments
    tabs = st.tabs([
        "📈 Aperçu Général",
        "🎯 Tous les Objectifs", 
        "📄 Objectifs des Documents",
        "🔍 Analyse de Contenu",
        "🏷️ Classification Bloom",
        "✨ Reformulation SMART",
        "⚖️ Évaluation Difficulté",
        "📚 Ressources Pédagogiques",
        "💡 Feedback & Conseils"
    ])
    
    # Onglet 1: Aperçu Général
    with tabs[0]:
        st.header("📈 Vue d'Ensemble")
        
        # Graphiques côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            bloom_chart = create_bloom_distribution_chart(objectives_with_levels)
            st.plotly_chart(bloom_chart, use_container_width=True, key="bloom_overview")
        
        with col2:
            difficulty_chart = create_difficulty_chart(objectives_with_difficulty)
            st.plotly_chart(difficulty_chart, use_container_width=True, key="difficulty_overview")
        
        # Résumé exécutif avec information sur les sources
        st.subheader("📋 Résumé Exécutif")
        
        # Affichage de la répartition des sources
        if document_objectives:
            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid #10b981;">
                <strong>🎯 Sources des Objectifs:</strong><br>
                • Contenu saisi: {len(content_objectives)} objectif(s)<br>
                • Documents uploadés: {len(document_objectives)} objectif(s)<br>
                • <strong>Total: {len(all_objectives)} objectif(s)</strong>
            </div>
            """, unsafe_allow_html=True)
        
        feedback_parts = feedback.split('\n\n')
        first_feedback = feedback_parts[0] if feedback_parts else feedback
        
        st.markdown(f"""
        <div class="result-card" style="border-left: 4px solid #4b6fff;">
            <strong>🎯 Analyse Principale:</strong><br><br>
            {first_feedback.replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Aperçu des objectifs les mieux formulés
        st.subheader("⭐ Exemples d'Objectifs Reformulés")
        formatted_list = formatted_objectives.split("\n\n")
        
        for i, formatted_obj in enumerate(formatted_list[:3], 1):
            if formatted_obj.strip():
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #10b981;">
                    <strong>Objectif #{i} (Format SMART):</strong><br><br>
                    {formatted_obj.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)
        
        if len(formatted_list) > 3:
            st.markdown("*Consultez l'onglet 'Reformulation SMART' pour voir tous les objectifs reformulés.*")
    
    # Onglet 2: Tous les Objectifs
    with tabs[1]:
        st.header("🎯 Tous les Objectifs d'Apprentissage")
        
        # Regroupement par niveau de Bloom
        bloom_groups = {}
        for obj, level in objectives_with_levels.items():
            if level not in bloom_groups:
                bloom_groups[level] = []
            
            # Déterminer la source de l'objectif avec gestion d'erreur
            source_type = "contenu"
            source_doc = "Contenu saisi"
            if document_objectives:
                for doc_obj in document_objectives:
                    try:
                        doc_objective_text = doc_obj.get('objective', '')
                        if doc_objective_text == obj:
                            source_type = "document"
                            source_doc = (doc_obj.get('source_document') or 
                                        doc_obj.get('document_source') or 
                                        doc_obj.get('source') or 
                                        'Document inconnu')
                            break
                    except Exception as e:
                        print(f"Erreur lors de la comparaison d'objectifs: {e}")
                        continue
            
            bloom_groups[level].append((obj, objectives_with_difficulty.get(obj, 3), source_type, source_doc))
        
        # Ordre et couleurs des niveaux de Bloom
        bloom_order = ["se souvenir", "comprendre", "appliquer", "analyser", "évaluer", "créer"]
        colors = {
            "se souvenir": "#ef4444", "comprendre": "#3b82f6", "appliquer": "#10b981",
            "analyser": "#f59e0b", "évaluer": "#8b5cf6", "créer": "#ec4899"
        }
        
        # Affichage par niveau de Bloom
        for level in bloom_order:
            if level in bloom_groups:
                color = colors.get(level, "#6b7280")
                st.markdown(f"""
                <h3 style="color: {color}; border-bottom: 2px solid {color}; padding-bottom: 0.5rem; margin-top: 2rem;">
                    🌸 {level.capitalize()} ({len(bloom_groups[level])} objectif{'s' if len(bloom_groups[level]) > 1 else ''})
                </h3>
                """, unsafe_allow_html=True)
                
                for i, (obj, difficulty, source_type, source_doc) in enumerate(bloom_groups[level], 1):
                    difficulty_colors = ["#3b82f6", "#10b981", "#f59e0b", "#f97316", "#ef4444"]
                    difficulty_color = difficulty_colors[min(difficulty-1, 4)]
                    
                    # Icône selon la source
                    source_icon = "📝" if source_type == "contenu" else "📄"
                    source_color = "#4b6fff" if source_type == "contenu" else "#10b981"
                    
                    st.markdown(f"""
                    <div class="objective-card" style="border-left-color: {color};">
                        <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 0.5rem;">
                            <strong style="flex-grow: 1;">#{i}. {obj}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                            <div style="display: flex; gap: 0.5rem; align-items: center;">
                                <span style="background-color: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {level}
                                </span>
                                <span style="background-color: {source_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {source_icon} {source_doc}
                                </span>
                            </div>
                            <span style="color: {difficulty_color}; font-weight: 600;">
                                Difficulté: {difficulty}/5
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
        # Affichage des objectifs non classifiés s'il y en a
        if "non classifié" in bloom_groups:
            st.markdown(f"""
            <h3 style="color: #6b7280; border-bottom: 2px solid #6b7280; padding-bottom: 0.5rem; margin-top: 2rem;">
                ❓ Non Classifiés ({len(bloom_groups['non classifié'])} objectif{'s' if len(bloom_groups['non classifié']) > 1 else ''})
            </h3>
            """, unsafe_allow_html=True)
            
            for i, (obj, difficulty, source_type, source_doc) in enumerate(bloom_groups["non classifié"], 1):
                difficulty_colors = ["#3b82f6", "#10b981", "#f59e0b", "#f97316", "#ef4444"]
                difficulty_color = difficulty_colors[min(difficulty-1, 4)]
                
                source_icon = "📝" if source_type == "contenu" else "📄"
                source_color = "#4b6fff" if source_type == "contenu" else "#10b981"
                
                st.markdown(f"""
                <div class="objective-card" style="border-left-color: #6b7280;">
                    <strong>#{i}. {obj}</strong>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem;">
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span style="background-color: #6b7280; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem;">
                                Non classifié
                            </span>
                            <span style="background-color: {source_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {source_icon} {source_doc}
                            </span>
                        </div>
                        <span style="color: {difficulty_color}; font-weight: 600;">
                            Difficulté: {difficulty}/5
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Onglet 3: Objectifs des Documents
    with tabs[2]:
        st.header("📄 Objectifs Extraits des Documents")
        
        if not document_objectives:
            st.info("📝 Aucun objectif d'apprentissage n'a été trouvé dans les documents uploadés.")
            st.markdown("""
            **Pourquoi cela peut arriver:**
            - Les documents ne contiennent pas d'objectifs explicites
            - Les objectifs sont formulés de manière non-standard
            - Les documents sont plutôt du contenu de cours que des plans pédagogiques
            
            **Suggestions:**
            - Uploadez des syllabus ou plans de cours
            - Incluez des documents avec des sections "Objectifs" ou "Compétences"
            - Vérifiez que vos documents sont bien pédagogiques
            """)
        else:
            st.success(f"🎉 {len(document_objectives)} objectif(s) d'apprentissage identifié(s) dans vos documents !")
            
            # Regrouper par document source avec gestion d'erreur
            docs_groups = {}
            for doc_obj in document_objectives:
                try:
                    # Essayer différentes clés possibles pour la source
                    source = (doc_obj.get('source_document') or 
                             doc_obj.get('document_source') or 
                             doc_obj.get('source') or 
                             'Document inconnu')
                    
                    if source not in docs_groups:
                        docs_groups[source] = []
                    docs_groups[source].append(doc_obj)
                except Exception as e:
                    print(f"Erreur lors du regroupement: {e}")
                    # Ajouter à un groupe par défaut en cas d'erreur
                    if 'Documents divers' not in docs_groups:
                        docs_groups['Documents divers'] = []
                    docs_groups['Documents divers'].append(doc_obj)
            
            # Affichage par document
            for doc_name, doc_objs in docs_groups.items():
                st.subheader(f"📄 {doc_name}")
                st.markdown(f"*{len(doc_objs)} objectif(s) trouvé(s)*")
                
                for i, doc_obj in enumerate(doc_objs, 1):
                    try:
                        # Gestion sécurisée de toutes les clés
                        objective_text = doc_obj.get('objective', 'Objectif non spécifié')
                        obj_type = doc_obj.get('type', 'non classifié')
                        source_text = doc_obj.get('source_text', doc_obj.get('source', 'Texte source non disponible'))
                        relevance_score = doc_obj.get('relevance_score', 0.0)
                        
                        # Déterminer la couleur selon le type
                        type_color = "#10b981" if obj_type == "explicite" else "#f59e0b"
                        type_icon = "🎯" if obj_type == "explicite" else "💡"
                        
                        st.markdown(f"""
                        <div class="result-card" style="border-left: 4px solid {type_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <strong>Objectif #{i}:</strong>
                                <span style="background-color: {type_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {type_icon} {obj_type.capitalize()}
                                </span>
                            </div>
                            <div style="font-size: 1.1rem; margin-bottom: 1rem; font-weight: 500;">
                                "{objective_text}"
                            </div>
                            <div style="background-color: #374151; padding: 0.75rem; border-radius: 6px; font-style: italic; font-size: 0.9rem;">
                                <strong>Extrait du document:</strong><br>
                                "{source_text[:200]}{'...' if len(source_text) > 200 else ''}"
                            </div>
                            <div style="text-align: right; margin-top: 0.5rem; font-size: 0.8rem; color: #9ca3af;">
                                Score de pertinence: {relevance_score:.3f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as obj_error:
                        # Affichage minimal en cas d'erreur
                        st.error(f"Erreur lors de l'affichage de l'objectif {i}: {obj_error}")
                        st.markdown(f"""
                        <div class="result-card" style="border-left: 4px solid #6b7280;">
                            <strong>⚠️ Objectif #{i}:</strong> Erreur d'affichage<br>
                            <small>Clés disponibles: {list(doc_obj.keys()) if isinstance(doc_obj, dict) else 'Format incorrect'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Onglet 4: Analyse de Contenu
    with tabs[3]:
        st.header("🔍 Analyse du Contenu Pédagogique")
        
        # Afficher si des documents ont été utilisés
        if st.session_state.processed_documents:
            st.info(f"📚 Analyse enrichie avec {len(st.session_state.processed_documents)} document(s) uploadé(s)")
        
        # Traiter l'analyse du contenu
        analysis_text = content_analysis
        
        # Essayer de diviser l'analyse en sections logiques
        if "1." in analysis_text or "**" in analysis_text:
            # Si l'analyse contient des sections numérotées ou formatées
            sections = re.split(r'\n(?=\d+\.|\*\*|\n)', analysis_text)
            sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]
        
        for i, section in enumerate(sections, 1):
                # Identifier le type de section basé sur le contenu
                section_lower = section.lower()
                
                if any(keyword in section_lower for keyword in ["sujet", "domaine", "thème", "principal"]):
                    icon = "🎯"
                    border_color = "#4b6fff"
                    title = "Sujet Principal"
                elif any(keyword in section_lower for keyword in ["niveau", "complexité", "difficulté"]):
                    icon = "📊"
                    border_color = "#f59e0b"
                    title = "Niveau de Complexité"
                elif any(keyword in section_lower for keyword in ["concept", "compétence", "connaissance", "clé"]):
                    icon = "🧠"
                    border_color = "#10b981"
                    title = "Concepts Clés"
                elif any(keyword in section_lower for keyword in ["prérequis", "préalable", "prerequisite"]):
                    icon = "📚"
                    border_color = "#8b5cf6"
                    title = "Prérequis"
                elif any(keyword in section_lower for keyword in ["point fort", "force", "lacune", "faiblesse"]):
                    icon = "⚖️"
                    border_color = "#ef4444"
                    title = "Points Forts et Lacunes"
                else:
                    icon = "💡"
                    border_color = "#6b7280"
                    title = f"Analyse - Section {i}"
                
                # Nettoyer le contenu de la section
                clean_section = section.replace('**', '').replace('*', '').strip()
                clean_section = re.sub(r'^\d+\.\s*', '', clean_section)  # Supprimer les numéros en début
                
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {border_color};">
                    <h4 style="color: {border_color}; margin-bottom: 1rem;">{icon} {title}</h4>
                    <div style="line-height: 1.6;">
                        {clean_section.replace('\n', '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Si pas de structure claire, diviser par paragraphes
            paragraphs = [p.strip() for p in analysis_text.split('\n\n') if p.strip()]
            
            if len(paragraphs) > 1:
                for i, paragraph in enumerate(paragraphs, 1):
                    if len(paragraph) > 30:  # Ignorer les paragraphes trop courts
                        # Déterminer le type basé sur les mots-clés
                        paragraph_lower = paragraph.lower()
                        
                        if any(keyword in paragraph_lower for keyword in ["sujet", "cours", "thème"]):
                            icon = "🎯"
                            border_color = "#4b6fff"
                            title = "Analyse du Sujet"
                        elif any(keyword in paragraph_lower for keyword in ["niveau", "difficulté", "complexe"]):
                            icon = "📊"
                            border_color = "#f59e0b"
                            title = "Niveau et Difficulté"
                        elif any(keyword in paragraph_lower for keyword in ["concept", "notion", "compétence"]):
                            icon = "🧠"
                            border_color = "#10b981"
                            title = "Concepts et Compétences"
                        elif any(keyword in paragraph_lower for keyword in ["prérequis", "préalable", "base"]):
                            icon = "📚"
                            border_color = "#8b5cf6"
                            title = "Prérequis et Bases"
                else:
                            icon = "💡"
                            border_color = "#6b7280"
                            title = f"Observation {i}"
                
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {border_color};">
                    <h4 style="color: {border_color}; margin-bottom: 1rem;">{icon} {title}</h4>
                            <div style="line-height: 1.6;">
                                {paragraph.replace('\n', '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Si un seul bloc, l'afficher tel quel
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #4b6fff;">
                    <h4 style="color: #4b6fff; margin-bottom: 1rem;">🔍 Analyse Complète</h4>
                    <div style="line-height: 1.6;">
                        {analysis_text.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Informations supplémentaires sur les documents utilisés
        if st.session_state.processed_documents:
            st.subheader("📄 Documents Analysés")
            
            cols = st.columns(min(len(st.session_state.processed_documents), 3))
            
            for i, doc in enumerate(st.session_state.processed_documents):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid #334155;">
                        <strong>📄 {doc['name']}</strong><br>
                        <small style="color: #9ca3af;">
                            {doc.get('word_count', 'N/A')} mots • 
                            {len(doc.get('text_preview', ''))} caractères
                        </small>
        </div>
                    """, unsafe_allow_html=True)
    
    # Onglet 5: Classification Bloom (CORRIGÉ)
    with tabs[4]:
        st.header("🏷️ Classification selon la Taxonomie de Bloom")
        
        # Affichage du graphique
        bloom_chart = create_bloom_distribution_chart(objectives_with_levels)
        st.plotly_chart(bloom_chart, use_container_width=True, key="bloom_classification")
        
        # Information sur la taxonomie
        st.subheader("🌸 À propos de la Taxonomie de Bloom")
        st.markdown("""
        <div class="result-card">
            La taxonomie de Bloom classe les objectifs d'apprentissage en 6 niveaux cognitifs, 
            du plus simple au plus complexe. Un bon cours devrait couvrir plusieurs niveaux 
            pour offrir une progression pédagogique complète.
        </div>
        """, unsafe_allow_html=True)
        
        # Détails de classification par objectif
        st.subheader("📋 Classification Détaillée")
        
        # Parser la classification pour extraire les informations par objectif
        classification_text = classification
        
        # Diviser la classification en sections par objectif
        if "Objectif:" in classification_text:
            # Pattern pour extraire chaque section d'objectif
            pattern = r"Objectif:\s*(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, classification_text, re.DOTALL)
            
            colors = {
                "se souvenir": "#ef4444", "comprendre": "#3b82f6", "appliquer": "#10b981",
                "analyser": "#f59e0b", "évaluer": "#8b5cf6", "créer": "#ec4899"
            }
            
            for i, (obj_text, obj_details) in enumerate(matches, 1):
                obj_text = obj_text.strip()
                level = objectives_with_levels.get(obj_text, "non classifié")
                color = colors.get(level, "#6b7280")
                
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <strong>Objectif #{i}:</strong>
                        <span style="background-color: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                            {level}
                        </span>
                    </div>
                    <div style="font-style: italic; margin-bottom: 1rem; padding: 0.5rem; background-color: #374151; border-radius: 6px;">
                        "{obj_text}"
                    </div>
                    <div style="line-height: 1.6;">
                        {obj_details.strip().replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Si le format est différent, afficher le texte brut
            st.markdown(f"""
            <div class="result-card">
                {classification_text.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
    
    # Onglet 6: Reformulation SMART (CORRIGÉ)
    with tabs[5]:
        st.header("✨ Objectifs Reformulés (Critères SMART)")
        
        # Information sur les critères SMART
        st.markdown("""
        <div class="result-card" style="border-left: 4px solid #4b6fff;">
            <strong>🎯 Critères SMART:</strong><br><br>
            <strong>S</strong>pécifique - Clairement défini et sans ambiguïté<br>
            <strong>M</strong>esurable - Avec des critères d'évaluation observables<br>
            <strong>A</strong>tteignable - Réaliste dans le contexte d'apprentissage<br>
            <strong>R</strong>elevant - Pertinent par rapport au domaine d'étude<br>
            <strong>T</strong>emporel - Avec une indication temporelle
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage des objectifs reformulés
        formatted_text = formatted_objectives
        
        # Essayer de diviser par objectifs distincts
        if "Objectif" in formatted_text or "À la fin" in formatted_text:
            # Diviser par lignes vides ou par patterns d'objectifs
            sections = re.split(r'\n\s*\n|(?=\d+\.|\*|-)|\bObjectif\s*\d+', formatted_text)
            sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]
            
            for i, section in enumerate(sections, 1):
                if section:
                    # Nettoyer la section
                    clean_section = section.replace('*', '').replace('#', '').strip()
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #10b981;">
                        <h4 style="color: #10b981; margin-bottom: 1rem;">✨ Objectif Reformulé #{i}</h4>
                        <div style="line-height: 1.6; font-size: 1.05rem;">
                            {clean_section.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Si pas de structure claire, diviser par paragraphes
            paragraphs = [p.strip() for p in formatted_text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs, 1):
                if len(paragraph) > 20:  # Ignorer les paragraphes trop courts
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #10b981;">
                        <h4 style="color: #10b981; margin-bottom: 1rem;">✨ Section {i}</h4>
                        <div style="line-height: 1.6;">
                            {paragraph.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Onglet 7: Évaluation de la Difficulté (CORRIGÉ)
    with tabs[6]:
        st.header("⚖️ Évaluation de la Difficulté")
        
        # Graphique de difficulté
        difficulty_chart = create_difficulty_chart(objectives_with_difficulty)
        st.plotly_chart(difficulty_chart, use_container_width=True, key="difficulty_detailed")
        
        # Légende des niveaux de difficulté
        st.subheader("📊 Échelle de Difficulté")
        st.markdown("""
        <div class="result-card">
            <div style="display: grid; gap: 0.75rem;">
                <div><span style="background-color: #3b82f6; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; margin-right: 1rem; font-weight: 600;">1 - Très facile</span> Connaissances de base, mémorisation simple</div>
                <div><span style="background-color: #10b981; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; margin-right: 1rem; font-weight: 600;">2 - Facile</span> Application simple, compréhension directe</div>
                <div><span style="background-color: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; margin-right: 1rem; font-weight: 600;">3 - Modéré</span> Analyse et application modérément complexe</div>
                <div><span style="background-color: #f97316; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; margin-right: 1rem; font-weight: 600;">4 - Difficile</span> Évaluation complexe, analyse approfondie</div>
                <div><span style="background-color: #ef4444; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; margin-right: 1rem; font-weight: 600;">5 - Très difficile</span> Création originale, synthèse complexe</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Évaluation détaillée par objectif
        st.subheader("🔍 Évaluation Détaillée par Objectif")
        
        difficulty_text = difficulty_evaluation
        
        # Parser l'évaluation de difficulté
        if "Objectif:" in difficulty_text:
            pattern = r"Objectif:\s*(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, difficulty_text, re.DOTALL)
            
            for i, (obj_text, obj_details) in enumerate(matches, 1):
                obj_text = obj_text.strip()
                difficulty = objectives_with_difficulty.get(obj_text, 3)
                difficulty_colors = ["#3b82f6", "#10b981", "#f59e0b", "#f97316", "#ef4444"]
                difficulty_color = difficulty_colors[min(difficulty-1, 4)]
                
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {difficulty_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <strong>Objectif #{i}:</strong>
                        <span style="background-color: {difficulty_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 600;">
                            Difficulté: {difficulty}/5
                        </span>
                    </div>
                    <div style="font-style: italic; margin-bottom: 1rem; padding: 0.5rem; background-color: #374151; border-radius: 6px;">
                        "{obj_text}"
                    </div>
                    <div style="line-height: 1.6;">
                        {obj_details.strip().replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Format alternatif
            sections = [s.strip() for s in difficulty_text.split('\n\n') if s.strip()]
            for i, section in enumerate(sections, 1):
                if len(section) > 20:
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #f59e0b;">
                        <h4 style="color: #f59e0b; margin-bottom: 1rem;">⚖️ Évaluation #{i}</h4>
                        <div style="line-height: 1.6;">
                            {section.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Onglet 8: Ressources Pédagogiques (CORRIGÉ)
    with tabs[7]:
        st.header("📚 Recommandations de Ressources Pédagogiques")
        
        # Introduction
        st.markdown("""
        <div class="result-card" style="border-left: 4px solid #8b5cf6;">
            <strong>💡 Guide d'utilisation:</strong><br>
            Ces recommandations sont adaptées à chaque objectif selon son niveau de Bloom et sa difficulté. 
            Elles incluent des activités d'apprentissage, des méthodes d'évaluation et des ressources génériques.
        </div>
        """, unsafe_allow_html=True)
        
        # Recommandations par objectif
        recommendations_text = recommendations
        
        colors = {
            "se souvenir": "#ef4444", "comprendre": "#3b82f6", "appliquer": "#10b981",
            "analyser": "#f59e0b", "évaluer": "#8b5cf6", "créer": "#ec4899"
        }
        
        # Parser les recommandations
        if "Objectif:" in recommendations_text:
            pattern = r"Objectif:\s*(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, recommendations_text, re.DOTALL)
            
            for i, (obj_text, obj_details) in enumerate(matches, 1):
                obj_text = obj_text.strip()
                level = objectives_with_levels.get(obj_text, "non classifié")
                color = colors.get(level, "#6b7280")
                
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <strong>Ressources pour l'Objectif #{i}:</strong>
                        <span style="background-color: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                            {level}
                        </span>
                    </div>
                    <div style="font-style: italic; margin-bottom: 1rem; padding: 0.5rem; background-color: #374151; border-radius: 6px;">
                        "{obj_text}"
                    </div>
                    <div style="line-height: 1.6;">
                        {obj_details.strip().replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Format par sections si pas de structure "Objectif:"
            sections = [s.strip() for s in recommendations_text.split('\n\n') if s.strip()]
            
            for i, section in enumerate(sections, 1):
                if len(section) > 30:  # Ignorer les sections trop courtes
                    # Essayer de déterminer de quel objectif il s'agit
                    matching_objective = None
                    for obj in all_objectives:
                        if obj.lower() in section.lower() or any(word in section.lower() for word in obj.lower().split()[:3]):
                            matching_objective = obj
                            break
                    
                    if matching_objective:
                        level = objectives_with_levels.get(matching_objective, "non classifié")
                        color = colors.get(level, "#6b7280")
                    else:
                        level = "général"
                        color = "#6b7280"
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid {color};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <strong>Recommandation #{i}:</strong>
                            <span style="background-color: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {level}
                            </span>
                        </div>
                        <div style="line-height: 1.6;">
                            {section.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Onglet 9: Feedback & Conseils (inchangé)
    with tabs[8]:
        st.header("💡 Feedback & Conseils")
        
        # Diviser le feedback en sections
        feedback_parts = feedback.split('\n\n')
        
        for i, part in enumerate(feedback_parts, 1):
            if part.strip():
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #4b6fff;">
                    <h4 style="color: #4b6fff; margin-bottom: 1rem;">💡 Section {i}</h4>
                    <div style="line-height: 1.6;">
                        {part.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Fonction principale de l'application"""
    
    # En-tête de l'application
    display_app_header()
    
    # Sidebar - Configuration et informations
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration de l'API
        api_key = st.text_input(
            "🔑 Clé API OpenAI",
            type="password",
            help="Votre clé API OpenAI pour utiliser les modèles d'IA",
            placeholder="sk-..."
        )
        
        # Sélection du modèle
        model = st.selectbox(
            "🤖 Modèle LLM",
            options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
            index=0,
            help="Modèle à utiliser pour l'analyse (gpt-3.5-turbo recommandé pour les coûts)"
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            temperature = st.slider(
                "🌡️ Température",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Contrôle la créativité des réponses (0 = déterministe, 1 = créatif)"
            )
            
            verbose = st.checkbox(
                "📝 Mode verbeux",
                value=True,
                help="Affiche les détails du processus dans la console"
            )
        
        st.divider()
        
        # Informations sur la session
        st.header("📊 Session Actuelle")
        
        session_info_container = st.container()
        with session_info_container:
            st.markdown(f"**ID de session:** `{st.session_state.session_id[:8]}...`")
            
            if st.session_state.processed_documents:
                st.success(f"📄 {len(st.session_state.processed_documents)} document(s) chargé(s)")
                
                with st.expander("📋 Détails des documents"):
                    for doc in st.session_state.processed_documents:
                        st.markdown(f"• **{doc['name']}**")
            else:
                st.info("📄 Aucun document chargé")
            
            if st.session_state.analysis_done:
                st.success("✅ Analyse terminée")
            else:
                st.info("⏳ En attente d'analyse")
        
        # Bouton pour nouvelle session
        if st.button("🔄 Nouvelle Session", use_container_width=True, help="Démarre une nouvelle session (efface tout)"):
            # Nettoyer l'ancienne session
            if st.session_state.agent:
                st.session_state.agent.clear_session()
            
            # Réinitialiser les variables de session
            st.session_state.agent = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.analysis_done = False
            st.session_state.form_data = {}
            st.session_state.processed_documents = []
            
            # Nettoyer les champs du formulaire
            for key in ["course_subject", "target_audience", "learning_objectives", "source_text"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("🔄 Nouvelle session créée!")
            st.rerun()
        
        st.divider()
        
        # Informations sur la taxonomie de Bloom
        with st.expander("🌸 Taxonomie de Bloom"):
            st.markdown("""
            **Les 6 niveaux cognitifs (du simple au complexe):**
            
            1. **Se souvenir** 🔴 - Récupérer des connaissances de la mémoire
            2. **Comprendre** 🔵 - Construire du sens à partir d'informations  
            3. **Appliquer** 🟢 - Utiliser des connaissances dans une situation
            4. **Analyser** 🟡 - Décomposer en parties et examiner les relations
            5. **Évaluer** 🟣 - Porter des jugements basés sur des critères
            6. **Créer** 🟡 - Assembler des éléments en un nouveau tout
            
            **💡 Conseil:** Un bon cours couvre plusieurs niveaux pour une progression complète.
            """)
        
        # Informations sur les critères SMART
        with st.expander("🎯 Critères SMART"):
            st.markdown("""
            **Pour des objectifs bien formulés:**
            
            - **S**pécifique - Clairement défini et sans ambiguïté
            - **M**esurable - Avec des critères d'évaluation observables
            - **A**tteignable - Réaliste dans le contexte d'apprentissage  
            - **R**elevant - Pertinent par rapport au domaine d'étude
            - **T**emporel - Avec une indication temporelle
            
            **Exemple:** "À la fin du module 3, l'étudiant sera capable de..."
            """)
        
        # Aide et support
        with st.expander("❓ Aide & Support"):
            st.markdown("""
            **🚀 Pour commencer:**
            1. Entrez votre clé API OpenAI
            2. (Optionnel) Téléchargez vos documents pédagogiques
            3. Remplissez au minimum le sujet de votre cours
            4. Cliquez sur "Analyser"
            
            **💡 Conseils pour de meilleurs résultats:**
            - Soyez précis dans la description du sujet
            - Incluez le niveau et les prérequis du public
            - Téléchargez des syllabus ou plans de cours
            - Mentionnez les compétences visées
            
            **⚠️ Problèmes courants:**
            - Erreur 429: Limite de tokens atteinte → Réduisez le contenu
            - Erreur 401: Clé API invalide → Vérifiez votre clé
            - Analyse vide: Contenu insuffisant → Ajoutez plus de détails
            """)
    
    # Initialisation de l'agent si clé API fournie
    if api_key and not st.session_state.agent:
        try:
            with st.spinner("🔄 Initialisation de l'agent IA..."):
                st.session_state.agent = EnhancedLearningObjectiveAgent(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    verbose=verbose
                )
            st.success("✅ Agent IA initialisé avec succès!")
            time.sleep(1)  # Petite pause pour que l'utilisateur voie le message
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation de l'agent: {str(e)}")
            st.markdown("""
            **Solutions possibles:**
            - Vérifiez que votre clé API OpenAI est correcte
            - Assurez-vous d'avoir un crédit suffisant sur votre compte OpenAI
            - Vérifiez votre connexion Internet
            - Réessayez dans quelques minutes
            """)
            return
    
    # Interface principale conditionnelle
    if not api_key:
        # Message d'accueil si pas de clé API
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; background-color: #1e293b; border-radius: 12px; margin: 2rem 0; border: 1px solid #334155;">
            <h2 style="color: #f9fafb; margin-bottom: 1rem;">🚀 Bienvenue dans l'Analyseur d'Objectifs d'Apprentissage</h2>
            <p style="color: #d1d5db; font-size: 1.1rem; margin-bottom: 2rem;">
                Commencez par entrer votre clé API OpenAI dans la barre latérale pour démarrer l'analyse.
            </p>
            <div style="background-color: #273549; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <h3 style="color: #4b6fff; margin-bottom: 1rem;">🎯 Fonctionnalités principales:</h3>
                <div style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <p>✨ <strong>Extraction automatique</strong> d'objectifs d'apprentissage</p>
                    <p>🏷️ <strong>Classification selon Bloom</strong> (6 niveaux cognitifs)</p>
                    <p>📝 <strong>Reformulation SMART</strong> des objectifs</p>
                    <p>⚖️ <strong>Évaluation de difficulté</strong> (échelle 1-5)</p>
                    <p>📚 <strong>Recommandations pédagogiques</strong> personnalisées</p>
                    <p>📄 <strong>Support de documents</strong> (PDF, Word, TXT)</p>
                    <p>📊 <strong>Visualisations interactives</strong> et métriques</p>
                    <p>💡 <strong>Feedback constructif</strong> et conseils d'amélioration</p>
                </div>
            </div>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                Besoin d'une clé API OpenAI? 
                <a href="https://platform.openai.com/api-keys" target="_blank" style="color: #4b6fff;">
                    Créez-en une ici
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    elif not st.session_state.agent:
        # En cours d'initialisation
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #1e293b; border-radius: 8px; margin: 2rem 0;">
            <div style="color: #f59e0b; font-size: 1.2rem;">⏳ Initialisation de l'agent en cours...</div>
            <p style="color: #d1d5db; margin-top: 1rem;">Veuillez patienter quelques instants.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Interface principale - Mode normal
    if not st.session_state.analysis_done:
        # Mode saisie: Gestion des documents + Formulaire
        
        # Section de gestion des documents
        display_document_manager()
        
        # Formulaire de saisie
        form_data = display_input_form()
        
        # Traitement des actions du formulaire
        if form_data["clear"]:
            # Effacer les champs du formulaire
            for key in ["course_subject", "target_audience", "learning_objectives", "source_text"]:
                if key in st.session_state:
                    st.session_state[key] = ""
            st.success("🧹 Formulaire effacé")
            st.rerun()
        
        if form_data["submit"]:
            # Validation des données obligatoires
            if not form_data["course_subject"].strip():
                st.error("❌ Le sujet du cours est obligatoire pour effectuer l'analyse.")
                st.info("💡 Décrivez au minimum le contenu principal de votre cours.")
                return
            
            # Préparation du contenu pour l'analyse
            content_parts = []
            
            # Ajout du contenu principal
            content_parts.append("# INFORMATIONS SUR LE COURS\n")
            
            content_parts.append(f"## Sujet Principal\n{form_data['course_subject']}\n")
            
            if form_data["target_audience"].strip():
                content_parts.append(f"## Public Cible\n{form_data['target_audience']}\n")
            
            if form_data["learning_objectives"].strip():
                content_parts.append(f"## Objectifs d'Apprentissage Existants\n{form_data['learning_objectives']}\n")
            
            if form_data["source_text"].strip():
                content_parts.append(f"## Contenu Supplémentaire\n{form_data['source_text']}\n")
            
            # Information sur les documents traités
            if st.session_state.processed_documents:
                content_parts.append(f"## Documents Pédagogiques Disponibles\n")
                content_parts.append("Les documents suivants ont été analysés et sont disponibles pour enrichir l'analyse:\n")
                for doc in st.session_state.processed_documents:
                    content_parts.append(f"- {doc['name']}\n")
            
            content = "\n".join(content_parts)
            
            # Sauvegarde des données du formulaire
            st.session_state.form_data = form_data
            
            # Lancement de l'analyse
            with st.spinner("🔍 Analyse en cours... Cela peut prendre 1-3 minutes selon la complexité."):
                try:
                    # Progress bar simulation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Étapes de progression
                    progress_bar.progress(10)
                    status_text.text("📚 Préparation du contenu...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(30)
                    status_text.text("🔍 Recherche dans les documents...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(50)
                    status_text.text("🎯 Extraction des objectifs...")
                    
                    # Exécution de l'analyse
                    results = st.session_state.agent.process_content_with_documents(content)
                    
                    progress_bar.progress(80)
                    status_text.text("🏷️ Classification et évaluation...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analyse terminée!")
                    time.sleep(0.5)
                    
                    # Nettoyage de l'interface de progression
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Sauvegarde des résultats
                    st.session_state.results = results
                    st.session_state.analysis_done = True
                    
                    # Sauvegarde sur disque
                    output_file = os.path.join(OUTPUT_DIRECTORY, f"analysis_{st.session_state.session_id[:8]}.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "session_id": st.session_state.session_id,
                            "timestamp": str(pd.Timestamp.now()),
                            "form_data": form_data,
                            "processed_documents": st.session_state.processed_documents,
                            "results": results
                        }, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ Analyse terminée! Résultats sauvegardés: {output_file}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite lors de l'analyse: {str(e)}")
                    st.markdown("""
                    **Solutions possibles:**
                    - Réduisez la quantité de contenu si elle est très importante
                    - Vérifiez votre clé API OpenAI et votre crédit
                    - Réessayez dans quelques minutes
                    - Contactez le support si le problème persiste
                    """)
                    
                    # Log de l'erreur pour débogage
                    error_log = {
                        "timestamp": str(pd.Timestamp.now()),
                        "session_id": st.session_state.session_id,
                        "error": str(e),
                        "content_length": len(content),
                        "model": model
                    }
                    
                    error_file = os.path.join(OUTPUT_DIRECTORY, f"error_{st.session_state.session_id[:8]}.json")
                    with open(error_file, "w", encoding="utf-8") as f:
                        json.dump(error_log, f, ensure_ascii=False, indent=2)
    
    else:
        # Mode résultats: Affichage des résultats de l'analyse
        display_results()

# Import nécessaire pour la fonction time.sleep
import time

# Point d'entrée de l'application
if __name__ == "__main__":
    main()