# Agent Learning Objective

Un agent d'intelligence artificielle qui utilise la taxonomie de Bloom pour analyser, classifier et améliorer les objectifs d'apprentissage.

## Fonctionnalités

- **Extraction d'objectifs**: Identifie les objectifs d'apprentissage explicites et implicites dans un texte
- **Analyse de contenu**: Évalue le sujet, la complexité et les concepts clés du contenu pédagogique
- **Classification selon Bloom**: Catégorise chaque objectif selon les six niveaux de la taxonomie de Bloom
- **Reformulation SMART**: Améliore les objectifs pour les rendre Spécifiques, Mesurables, Atteignables, Pertinents et Temporellement définis
- **Évaluation de difficulté**: Détermine le niveau de difficulté de chaque objectif
- **Recommandations**: Suggère des activités, méthodes d'évaluation et ressources adaptées
- **Feedback constructif**: Fournit une analyse critique de l'ensemble des objectifs
- **Interface utilisateur moderne**: Interface Streamlit intuitive avec visualisations interactives
- **Gestion de documents**: Support pour l'upload et l'analyse de documents PDF, TXT et DOCX
- **Recherche sémantique**: Recherche intelligente dans les documents chargés
- **Stockage vectoriel**: Utilisation de Pinecone pour l'indexation et la recherche de contenu

## Structure du Projet

```
learning-objective-agent/
├── streamlit_app2.py      # Application principale Streamlit
├── enhanced_agent.py      # Agent amélioré avec gestion de documents
├── components.py          # Composants de traitement (extracteur, analyseur, etc.)
├── config.py             # Configuration (API keys, paramètres)
├── bloom_taxonomy.py     # Définitions de la taxonomie de Bloom
├── requirements.txt      # Dépendances Python
├── outputs/             # Dossier pour les résultats générés
└── README.md           # Documentation
```

## Installation

1. Clonez ce dépôt:
```bash
git clone https://github.com/IlyasOuchen1/learning-obj-agent2.git
cd learning-obj-agent2
```

2. Créez et activez un environnement virtuel:
```bash
python -m venv .venv
# Sur Windows
.venv\Scripts\activate
# Sur Linux/Mac
source .venv/bin/activate
```

3. Installez les dépendances:
```bash
pip install -r requirements.txt
```

4. Configurez vos clés API:
   - Créez un fichier `config.py` avec vos clés API
   - Ou définissez les variables d'environnement:
```bash
# Sur Windows
set OPENAI_API_KEY=votre-clé-api
set PINECONE_API_KEY=votre-clé-pinecone
# Sur Linux/Mac
export OPENAI_API_KEY=votre-clé-api
export PINECONE_API_KEY=votre-clé-pinecone
```

## Utilisation

### Interface Web (Recommandée)

1. Lancez l'application Streamlit:
```bash
streamlit run streamlit_app2.py
```

2. Accédez à l'interface web dans votre navigateur (généralement http://localhost:8501)
3. Vous pouvez:
   - Entrer du texte directement dans l'interface
   - Uploader des documents (PDF, TXT, DOCX)
   - Rechercher dans les documents chargés
   - Visualiser les résultats avec les graphiques interactifs
   -Entrez votre contenu pédagogique dans l'interface
   -Visualisez les résultats avec les graphiques interactifs

### Utilisation Programmatique

```python
from enhanced_agent import EnhancedLearningObjectiveAgent

# Initialisation de l'agent
agent = EnhancedLearningObjectiveAgent(api_key="votre-clé-api")

# Traitement du contenu
content = """
Ce cours d'introduction à l'intelligence artificielle couvrira les fondamentaux de l'IA.
Les étudiants devront implémenter des modèles simples et analyser leurs performances.
"""

results = agent.process_content(content)

# Traitement de documents
files = ["document1.pdf", "document2.docx"]
session_id = "unique_session_id"
results = agent.process_uploaded_files(files, session_id)

# Accès aux résultats
objectives = results["objectives"]
classification = results["classification"]["classification"]
formatted_objectives = results["formatted_objectives"]["formatted_objectives"]
```

## Fonctionnalités de l'Interface

- **Interface moderne**: Design sombre avec une mise en page claire
- **Gestion de documents**: 
  - Upload multiple de fichiers
  - Support PDF, TXT, DOCX
  - Recherche sémantique
- **Visualisations interactives**: 
  - Distribution des niveaux de Bloom
  - Niveaux de difficulté
  - Métriques clés
- **Formulaires intuitifs**: 
  - Saisie de texte avec validation
  - Options de configuration
  - Téléchargement de fichiers
- **Résultats détaillés**:
  - Objectifs extraits et classifiés
  - Feedback constructif
  - Recommandations de ressources

## Personnalisation

Vous pouvez personnaliser:
- Les prompts des composants dans `components.py`
- Les paramètres de l'agent dans `config.py`
- L'interface utilisateur dans `streamlit_app2.py`
- La taxonomie de Bloom dans `bloom_taxonomy.py`
- Les paramètres de stockage vectoriel dans `enhanced_agent.py`

## Contribuer

Les contributions sont les bienvenues! N'hésitez pas à:
1. Ouvrir une issue pour signaler un bug ou proposer une amélioration
2. Soumettre une pull request avec vos modifications
3. Améliorer la documentation

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.