import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
import json
from agent import LearningObjectiveAgent

# Importer la configuration
try:
    from config import OPENAI_API_KEY, LLM_MODEL, OUTPUT_DIRECTORY
except ImportError:
    # Fallback si config.py n'est pas disponible
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    OUTPUT_DIRECTORY = os.environ.get("OUTPUT_DIRECTORY", "./outputs")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Configuration de la page
st.set_page_config(
    page_title="Learning Objective Generator",
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
    
    /* En-tête de la carte */
    .form-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #334155;
    }
    
    .form-icon {
        background-color: #1e40af;
        color: #ffffff;
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        margin-right: 1rem;
    }
    
    .form-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f9fafb !important;
        margin: 0;
        flex-grow: 1;
    }
    
    .required-label {
        background-color: #991b1b;
        color: #fecaca;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .optional-label {
        background-color: #374151;
        color: #d1d5db;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Corps de la carte */
    .form-body {
        padding: 0.5rem 0;
    }
    
    /* Champ de texte */
    .form-field {
        margin-bottom: 1rem;
    }
    
    .form-field-label {
        font-weight: 500;
        margin-bottom: 0.5rem;
        color: #f9fafb !important;
    }
    
    .form-field-help {
        font-size: 0.875rem;
        color: #9ca3af !important;
        margin-top: 0.25rem;
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
    
    /* Zone de téléchargement de fichiers */
    .file-uploader {
        border: 2px dashed #4b5563 !important;
        border-radius: 8px;
        padding: 2rem 1rem;
        text-align: center;
        background-color: #273549 !important;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #f9fafb !important;
    }
    
    .file-uploader:hover {
        border-color: #4b6fff !important;
        background-color: #1e293b !important;
    }
    
    /* Boutons */
    .button-container {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
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
    
    /* Objectifs par niveau de difficulté - version sombre */
    .difficulty-card {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        background-color: #1e293b !important;
        color: #f9fafb !important;
    }
    
    .difficulty-1 { border-left: 5px solid #1890ff; }
    .difficulty-2 { border-left: 5px solid #13c2c2; }
    .difficulty-3 { border-left: 5px solid #fadb14; }
    .difficulty-4 { border-left: 5px solid #fa8c16; }
    .difficulty-5 { border-left: 5px solid #f5222d; }
    
    /* Objectifs reformulés - version sombre */
    .smart-objective {
        background-color: #1e293b !important;
        border-left: 4px solid #4b6fff;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: #f9fafb !important;
    }
    
    /* Cartes de feedback - version sombre */
    .feedback-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1.25rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: #f9fafb !important;
    }
    
    /* Spinner de chargement */
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    /* Carte de métrique - version sombre */
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
    
    /* Pour le retour au formulaire */
    .back-button {
        margin-bottom: 1.5rem;
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
    
    /* Correction pour les containers génériques avec fond blanc */
    div[data-testid="stHorizontalBlock"] > div,
    div[data-testid="stVerticalBlock"] > div,
    div[class*="block-container"],
    .element-container > div {
        background-color: transparent !important;
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
</style>
""", unsafe_allow_html=True)

# Fonction pour extraire le niveau de difficulté
def extract_difficulty_level(difficulty_text, objective):
    pattern = re.compile(rf"{re.escape(objective)}.*?niveau de difficulté.*?(\d)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(difficulty_text)
    if match:
        return int(match.group(1))
    return 3  # Niveau moyen par défaut

# Fonction pour extraire le niveau de Bloom
def extract_bloom_level(classification_text, objective):
    # Pattern pour trouver l'objectif et son niveau associé
    pattern = re.compile(rf"{re.escape(objective)}.*?Niveau de Bloom: ([a-zéè ]+)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(classification_text)
    if match:
        level = match.group(1).strip().lower()
        return level
    return "non classifié"

# Fonction pour créer la visualisation de la distribution des niveaux Bloom
def create_bloom_distribution_chart(objectives_with_levels):
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

# Fonction pour créer le graphique de difficulté
def create_difficulty_chart(objectives_with_difficulty):
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

# Fonction pour afficher un badge Bloom
def bloom_badge(level):
    level_class = level.lower().replace(" ", "-").replace("é", "e").replace("è", "e")
    return f'<span class="bloom-badge {level_class}">{level.capitalize()}</span>'

# En-tête de l'application
def display_app_header():
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Générer une analyse d'objectifs d'apprentissage</h1>
        <p class="app-description">Fournissez des informations sur votre cours ou votre contenu pédagogique pour obtenir une analyse détaillée selon la taxonomie de Bloom. Plus vous fournissez de détails, meilleure sera l'analyse.</p>
    </div>
    """, unsafe_allow_html=True)

# Formulaire de saisie d'informations structuré
def display_input_form():
    with st.container():
        st.markdown("""
        ## Générer une analyse d'objectifs d'apprentissage
        
        Fournissez des informations sur votre cours ou votre contenu pédagogique pour obtenir une analyse détaillée selon la taxonomie de Bloom.
        Plus vous fournissez de détails, meilleure sera l'analyse.
        """)
        
        # Section 1: Sujet du cours
        with st.container():
            st.markdown("""
            <div class="form-card">
                <div class="form-header">
                    <div class="form-icon">💡</div>
                    <h3 class="form-title">De quoi traite votre cours?</h3>
                    <span class="required-label">Requis</span>
                </div>
                <div class="form-body">
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="stTextInput-wrapper">', unsafe_allow_html=True)
            course_subject = st.text_area(
                "Décrivez le sujet principal et le contenu général de votre cours",
                height=120, 
                placeholder="Exemple: Ce cours porte sur les principes fondamentaux de l'intelligence artificielle, couvrant l'apprentissage automatique, les réseaux de neurones et les applications pratiques de l'IA.",
                key="course_subject",
                label_visibility="visible"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
                        <p class="form-field-help">Soyez aussi précis que possible sur le sujet, les concepts clés, et les compétences visées.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 2: Public cible
        with st.container():
            st.markdown("""
            <div class="form-card">
                <div class="form-header">
                    <div class="form-icon">👥</div>
                    <h3 class="form-title">Si vous avez un public cible, qui sont-ils?</h3>
                    <span class="optional-label">Optionnel</span>
                </div>
                <div class="form-body">
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="stTextInput-wrapper">', unsafe_allow_html=True)
            target_audience = st.text_area(
                "Décrivez votre public cible (niveau, prérequis, contexte, etc.)",
                height=100,
                placeholder="Exemple: Étudiants de niveau licence en informatique ayant des connaissances de base en programmation Python et en mathématiques (algèbre linéaire et statistiques).",
                key="target_audience",
                label_visibility="visible"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
                        <p class="form-field-help">Le niveau et les connaissances préalables de votre public influenceront la difficulté et la formulation des objectifs.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 3: Objectifs d'apprentissage
        with st.container():
            st.markdown("""
            <div class="form-card">
                <div class="form-header">
                    <div class="form-icon">🎯</div>
                    <h3 class="form-title">Si vous avez des objectifs d'apprentissage spécifiques, quels sont-ils?</h3>
                    <span class="optional-label">Optionnel</span>
                </div>
                <div class="form-body">
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="stTextInput-wrapper">', unsafe_allow_html=True)
            learning_objectives = st.text_area(
                "Listez vos objectifs d'apprentissage actuels (un par ligne)",
                height=150,
                placeholder="Exemple:\n- Comprendre les principes fondamentaux de l'IA\n- Implémenter des algorithmes d'apprentissage automatique simples\n- Analyser les performances des modèles d'IA\n- Évaluer les implications éthiques de l'IA",
                key="learning_objectives",
                label_visibility="visible"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
                        <p class="form-field-help">Si vous n'avez pas encore d'objectifs définis, laissez ce champ vide. L'IA les générera à partir des autres informations.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 4: Contenu source
        with st.container():
            st.markdown("""
            <div class="form-card">
                <div class="form-header">
                    <div class="form-icon">📄</div>
                    <h3 class="form-title">Quel contenu source dois-je référencer? (L'ajout de contenu améliorera nos résultats)</h3>
                    <span class="optional-label">Optionnel</span>
                </div>
                <div class="form-body">
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="stTextInput-wrapper">', unsafe_allow_html=True)
                source_files = st.file_uploader("Glissez-déposez des fichiers ici ou choisissez un fichier", 
                                              accept_multiple_files=True,
                                              type=["docx", "pdf", "txt", "pptx"])
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="form-field-help">Téléchargez des plans de cours, des syllabus ou d\'autres documents pertinents.</p>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stTextInput-wrapper">', unsafe_allow_html=True)
                source_text = st.text_area(
                    "Collez le texte que vous souhaitez que je référence",
                    height=150,
                    placeholder="Collez ici du contenu supplémentaire comme un plan de cours, des descriptions de modules, etc.",
                    key="source_text",
                    label_visibility="visible"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="form-field-help">Plus vous fournissez de contenu, plus l\'analyse sera précise et personnalisée.</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Boutons d'action
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            submit_button = st.button("Analyser", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Effacer", use_container_width=True)
        with col3:
            st.markdown('<p class="caption">L\'analyse peut prendre jusqu\'à une minute selon la quantité de contenu fournie.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        "course_subject": course_subject,
        "target_audience": target_audience,
        "learning_objectives": learning_objectives,
        "source_files": source_files,
        "source_text": source_text,
        "submit": submit_button,
        "clear": clear_button
    }

# Fonction pour afficher les métriques
def display_metrics(num_objectives, highest_level, avg_difficulty):
    st.markdown('<div style="display: flex; gap: 1rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Nombre d'objectifs</div>
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fonction pour obtenir une couleur en fonction du niveau de Bloom
def get_color_for_level(level):
    level = level.lower().strip()
    colors = {
        "se souvenir": "#ef4444",
        "comprendre": "#3b82f6",
        "appliquer": "#10b981",
        "analyser": "#f59e0b",
        "évaluer": "#8b5cf6",
        "créer": "#ec4899",
        "non classifié": "#6b7280"
    }
    return colors.get(level, "#6b7280")

# Fonction principale pour l'interface
def main():
    # En-tête de l'application
    display_app_header()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input("Clé API OpenAI", value=OPENAI_API_KEY if OPENAI_API_KEY else "", type="password")
        model = st.selectbox("Modèle LLM", 
                            options=["gpt-4.1-mini", "gpt-4", "gpt-3.5-turbo"], 
                            index=0 if LLM_MODEL == "gpt-4.1-mini" else (1 if LLM_MODEL == "gpt-4" else 2))
        
        temperature = st.slider("Température", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                              help="Contrôle la créativité des réponses (0 = déterministe, 1 = créatif)")
        
        verbose = st.checkbox("Mode verbeux", value=True, 
                            help="Affiche les détails du processus dans la console")
        
        st.divider()
        
        st.header("À propos")
        st.markdown("""
        Cet outil utilise l'IA pour analyser et améliorer les objectifs d'apprentissage selon la taxonomie de Bloom.
        
        **Fonctionnalités:**
        - Extraction d'objectifs
        - Classification selon Bloom
        - Reformulation SMART
        - Évaluation de difficulté
        - Recommandations pédagogiques
        """)
        
        st.divider()
        
        # Informations sur la taxonomie de Bloom
        with st.expander("Taxonomie de Bloom"):
            st.markdown("""
            **Les 6 niveaux de la taxonomie de Bloom:**
            
            1. **Se souvenir** <span style="color:#ef4444">●</span> - Récupérer des connaissances de la mémoire
            2. **Comprendre** <span style="color:#3b82f6">●</span> - Construire du sens à partir de messages
            3. **Appliquer** <span style="color:#10b981">●</span> - Exécuter une procédure dans une situation donnée
            4. **Analyser** <span style="color:#f59e0b">●</span> - Décomposer un matériel en ses parties constitutives
            5. **Évaluer** <span style="color:#8b5cf6">●</span> - Porter des jugements basés sur des critères
            6. **Créer** <span style="color:#ec4899">●</span> - Assembler des éléments pour former un tout cohérent
            """, unsafe_allow_html=True)
    
    # Vérification de l'état de la session
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Affichage du formulaire ou des résultats
    if not st.session_state.analysis_done:
        # Affichage du formulaire d'entrée
        form_data = display_input_form()
        
        # Si le bouton Effacer est cliqué
        if form_data["clear"]:
            # Réinitialiser les valeurs du formulaire
            for key in st.session_state.keys():
                if key in ["course_subject", "target_audience", "learning_objectives", "source_text"]:
                    st.session_state[key] = ""
            
            st.session_state.form_data = {}
            st.rerun()
        
        # Si le bouton Analyser est cliqué
        if form_data["submit"]:
            # Vérifier que le sujet du cours est rempli
            if not form_data["course_subject"]:
                st.error("Veuillez décrire le sujet de votre cours (champ requis)")
                st.stop()
                
            # Vérification de la clé API
            if not api_key:
                st.error("Veuillez entrer une clé API OpenAI dans la barre latérale")
                st.stop()
            
            # Préparation du contenu à analyser
            content = "# Informations sur le cours\n\n"
            content += f"## Sujet du cours\n{form_data['course_subject']}\n\n"
            
            if form_data["target_audience"]:
                content += f"## Public cible\n{form_data['target_audience']}\n\n"
            
            if form_data["learning_objectives"]:
                content += f"## Objectifs d'apprentissage actuels\n{form_data['learning_objectives']}\n\n"
            
            if form_data["source_text"]:
                content += f"## Contenu supplémentaire\n{form_data['source_text']}\n\n"
            
            # Traitement des fichiers uploadés
            if form_data["source_files"]:
                content += "## Fichiers sources\n"
                for file in form_data["source_files"]:
                    # Stocker le contenu du fichier dans la session pour l'utiliser plus tard
                    file_content = file.read().decode('utf-8', errors='ignore')
                    content += f"### {file.name}\n{file_content}\n\n"
            
            # Sauvegarde des données du formulaire dans la session
            st.session_state.form_data = form_data
            
            # Affichage d'un spinner pendant le traitement
            with st.spinner("🔍 Analyse en cours... Cela peut prendre jusqu'à une minute."):
                try:
                    # Initialisation de l'agent
                    agent = LearningObjectiveAgent(
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        verbose=verbose
                    )
                    
                    # Traitement du contenu
                    results = agent.process_content(content)
                    
                    # Sauvegarde des résultats dans la session
                    st.session_state.results = results
                    st.session_state.analysis_done = True
                    
                    # Sauvegarde des résultats
                    output_file = os.path.join(OUTPUT_DIRECTORY, "last_analysis.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'analyse: {str(e)}")
                    st.write("Veuillez vérifier votre clé API et réessayer.")
    
    else:
        # Affichage des résultats de l'analyse
        results = st.session_state.results
        
        # Bouton pour revenir au formulaire
        col1, col2 = st.columns([1, 7])
        with col1:
            st.markdown('<div class="back-button">', unsafe_allow_html=True)
            if st.button("← Retour", type="primary"):
                st.session_state.analysis_done = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Vérification des erreurs dans les résultats
        if "error" in results and results["error"]:
            st.error(f"Une erreur s'est produite lors de l'analyse: {results['error']}")
            st.button("Retour au formulaire", on_click=lambda: setattr(st.session_state, "analysis_done", False))
            st.stop()
        
        # Extraction des résultats
        objectives = results["objectives"]
        content_analysis = results["content_analysis"]["analysis"]
        classification = results["classification"]["classification"]
        formatted_objectives = results["formatted_objectives"]["formatted_objectives"]
        difficulty_evaluation = results["difficulty_evaluation"]["difficulty_evaluation"]
        recommendations = results["recommendations"]["recommendations"]
        feedback = results["feedback"]["feedback"]
        
        # Extraction des niveaux de Bloom pour chaque objectif
        objectives_with_levels = {}
        for obj in objectives:
            bloom_level = extract_bloom_level(classification, obj)
            objectives_with_levels[obj] = bloom_level
        
        # Extraction des niveaux de difficulté
        objectives_with_difficulty = {}
        for obj in objectives:
            difficulty_level = extract_difficulty_level(difficulty_evaluation, obj)
            objectives_with_difficulty[obj] = difficulty_level
            
        # Onglets pour les résultats avec des icônes
        results_tabs = st.tabs([
            "📊 Aperçu", 
            "🎯 Objectifs", 
            "🔍 Analyse", 
            "🧩 Classification", 
            "📝 Reformulation", 
            "⚖️ Difficulté",
            "📚 Ressources", 
            "💬 Feedback"
        ])
        
        # Onglet Aperçu
        with results_tabs[0]:
            st.header("Aperçu des résultats")
            
            # Affichage des métriques principales
            # Comptage des niveaux
            bloom_levels_count = {}
            for level in objectives_with_levels.values():
                if level in bloom_levels_count:
                    bloom_levels_count[level] += 1
                else:
                    bloom_levels_count[level] = 1
            
            highest_level = max(bloom_levels_count.items(), key=lambda x: x[1])[0] if bloom_levels_count else "Non disponible"
            
            # Moyenne de difficulté
            avg_difficulty = sum(objectives_with_difficulty.values()) / len(objectives_with_difficulty) if objectives_with_difficulty else 0
            
            # Affichage des métriques
            display_metrics(len(objectives), highest_level, avg_difficulty)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique de distribution des niveaux Bloom
                bloom_chart = create_bloom_distribution_chart(objectives_with_levels)
                st.plotly_chart(bloom_chart, use_container_width=True, key="bloom_chart_overview")
            
            with col2:
                # Graphique de difficulté
                difficulty_chart = create_difficulty_chart(objectives_with_difficulty)
                st.plotly_chart(difficulty_chart, use_container_width=True, key="difficulty_chart_overview")
            
            # Analyse rapide
            st.subheader("Analyse rapide")
            st.markdown(f"""
            <div class="feedback-card">
                {feedback.split('\n\n')[0] if '\n\n' in feedback else feedback}
            </div>
            """, unsafe_allow_html=True)
            
            # Affichage des 3 premiers objectifs reformulés
            st.subheader("Exemple d'objectifs reformulés")
            formatted_list = formatted_objectives.split("\n\n")
            for i, formatted_obj in enumerate(formatted_list[:3]):
                if formatted_obj.strip():
                    st.markdown(f"""
                    <div class="smart-objective">
                        {formatted_obj.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(formatted_list) > 3:
                st.markdown("*Voir plus d'objectifs reformulés dans l'onglet 'Reformulation'*")
        
        # Onglet Objectifs
        with results_tabs[1]:
            st.header("Objectifs d'apprentissage extraits")
            
            # Regrouper les objectifs par niveau de Bloom
            bloom_groups = {}
            for obj, level in objectives_with_levels.items():
                if level not in bloom_groups:
                    bloom_groups[level] = []
                bloom_groups[level].append((obj, objectives_with_difficulty.get(obj, 3)))
            
            # Ordre des niveaux de Bloom
            bloom_order = ["se souvenir", "comprendre", "appliquer", "analyser", "évaluer", "créer"]
            
            # Afficher les objectifs par niveau
            for level in bloom_order:
                if level in bloom_groups:
                    st.subheader(f"{level.capitalize()} ({len(bloom_groups[level])})")
                    for obj, difficulty in bloom_groups[level]:
                        st.markdown(f"""
                        <div class="difficulty-card difficulty-{difficulty}">
                            <strong>{obj}</strong>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                {bloom_badge(level)} <span>Difficulté: {difficulty}/5</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Afficher les objectifs non classifiés s'il y en a
            if "non classifié" in bloom_groups:
                st.subheader(f"Non classifiés ({len(bloom_groups['non classifié'])})")
                for obj, difficulty in bloom_groups["non classifié"]:
                    st.markdown(f"""
                    <div class="difficulty-card difficulty-{difficulty}">
                        <strong>{obj}</strong>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <span>Non classifié</span> <span>Difficulté: {difficulty}/5</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Onglet Analyse
        with results_tabs[2]:
            st.header("Analyse du contenu pédagogique")
            
            # Découper l'analyse en sections si possible
            sections = content_analysis.split("\n\n")
            
            for section in sections:
                if section.strip():
                    st.markdown(f"""
                    <div class="feedback-card">
                        {section.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Onglet Classification
        with results_tabs[3]:
            st.header("Classification selon la taxonomie de Bloom")
            
            # Visualisation de la distribution
            bloom_chart = create_bloom_distribution_chart(objectives_with_levels)
            st.plotly_chart(bloom_chart, use_container_width=True, key="bloom_chart_classification")
            
            # Affichage de la classification détaillée
            # Découper la classification en sections par objectif
            pattern = r"Objectif:\s+(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, classification, re.DOTALL)
            
            for obj_text, obj_details in matches:
                st.markdown(f"""
                <div class="feedback-card">
                    <strong>Objectif:</strong> {obj_text.strip()}<br>
                    {obj_details.strip().replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
        
        # Onglet Reformulation
        with results_tabs[4]:
            st.header("Reformulation SMART des objectifs")
            
            # Séparation en objectifs individuels
            formatted_list = formatted_objectives.split("\n\n")
            
            for i, formatted_obj in enumerate(formatted_list, 1):
                if formatted_obj.strip():
                    st.markdown(f"""
                    <div class="smart-objective">
                        <strong>Objectif {i}</strong><br><br>
                        {formatted_obj.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Onglet Difficulté
        with results_tabs[5]:
            st.header("Évaluation de la difficulté")
            
            # Visualisation de la difficulté
            difficulty_chart = create_difficulty_chart(objectives_with_difficulty)
            st.plotly_chart(difficulty_chart, use_container_width=True, key="difficulty_chart_difficulty")
            
            # Explications des niveaux de difficulté
            st.markdown("""
            <div class="feedback-card">
                <strong>Échelle de difficulté:</strong><br>
                <span class="bloom-badge" style="background-color: #1e40af; color: #bfdbfe;">1 - Très facile</span> - Connaissances de base, mémorisation simple<br>
                <span class="bloom-badge" style="background-color: #065f46; color: #a7f3d0;">2 - Facile</span> - Application simple, compréhension directe<br>
                <span class="bloom-badge" style="background-color: #92400e; color: #fde68a;">3 - Modéré</span> - Analyse et application modérément complexe<br>
                <span class="bloom-badge" style="background-color: #c2410c; color: #fed7aa;">4 - Difficile</span> - Évaluation complexe, analyse approfondie<br>
                <span class="bloom-badge" style="background-color: #b91c1c; color: #fecaca;">5 - Très difficile</span> - Création originale, synthèse complexe
            </div>
            """, unsafe_allow_html=True)
            
            # Affichage de l'évaluation détaillée
            # Découper l'évaluation en sections par objectif
            pattern = r"Objectif:\s+(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, difficulty_evaluation, re.DOTALL)
            
            for obj_text, obj_details in matches:
                difficulty = objectives_with_difficulty.get(obj_text.strip(), 3)
                st.markdown(f"""
                <div class="difficulty-card difficulty-{difficulty}">
                    <strong>Objectif:</strong> {obj_text.strip()}<br><br>
                    {obj_details.strip().replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
        
        # Onglet Ressources
        with results_tabs[6]:
            st.header("Recommandations de ressources pédagogiques")
            
            # Découper les recommandations en sections par objectif
            pattern = r"Objectif:\s+(.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, recommendations, re.DOTALL)
            
            for obj_text, obj_details in matches:
                level = objectives_with_levels.get(obj_text.strip(), "non classifié")
                level_class = level.lower().replace(" ", "-").replace("é", "e").replace("è", "e")
                
                st.markdown(f"""
                <div class="feedback-card" style="border-left: 5px solid {get_color_for_level(level)}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <strong>Objectif:</strong> {obj_text.strip()}
                        {bloom_badge(level)}
                    </div>
                    {obj_details.strip().replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
        
        # Onglet Feedback
        with results_tabs[7]:
            st.header("Feedback général sur les objectifs d'apprentissage")
            
            # Découper le feedback en sections
            sections = feedback.split("\n\n")
            
            for section in sections:
                if section.strip():
                    st.markdown(f"""
                    <div class="feedback-card">
                        {section.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Option pour télécharger les résultats
            st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.download_button(
                    label="📥 Télécharger les résultats",
                    data=json.dumps(results, ensure_ascii=False, indent=2),
                    file_name="analyse_objectifs.json",
                    mime="application/json",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

# Exécution de l'application
if __name__ == "__main__":
    main()