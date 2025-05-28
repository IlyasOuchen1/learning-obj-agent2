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

## Structure du Projet

```
learning-objective-agent/
├── streamlit_app2.py      # Application principale Streamlit
├── agent.py              # Logique principale de l'agent
├── components.py         # Composants de traitement (extracteur, analyseur, etc.)
├── config.py            # Configuration (API keys, paramètres)
├── bloom_taxonomy.py    # Définitions de la taxonomie de Bloom
├── requirements.txt     # Dépendances Python
├── outputs/            # Dossier pour les résultats générés
└── README.md          # Documentation
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

4. Configurez votre clé API OpenAI:
   - Créez un fichier `config.py` avec votre clé API
   - Ou définissez la variable d'environnement:
```bash
# Sur Windows
set OPENAI_API_KEY=votre-clé-api
# Sur Linux/Mac
export OPENAI_API_KEY=votre-clé-api
```

## Utilisation

### Interface Web (Recommandée)

1. Lancez l'application Streamlit:
```bash
streamlit run streamlit_app2.py
```

2. Accédez à l'interface web dans votre navigateur (généralement http://localhost:8501)
3. Entrez votre contenu pédagogique dans l'interface
4. Visualisez les résultats avec les graphiques interactifs

### Utilisation Programmatique

```python
from agent import LearningObjectiveAgent

# Initialisation de l'agent
agent = LearningObjectiveAgent(api_key="votre-clé-api")

# Traitement du contenu
content = """
Ce cours d'introduction à l'intelligence artificielle couvrira les fondamentaux de l'IA.
Les étudiants devront implémenter des modèles simples et analyser leurs performances.
"""

results = agent.process_content(content)

# Accès aux résultats
objectives = results["objectives"]
classification = results["classification"]["classification"]
formatted_objectives = results["formatted_objectives"]["formatted_objectives"]
```

## Fonctionnalités de l'Interface

- **Interface moderne**: Design sombre avec une mise en page claire
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

## Contribuer

Les contributions sont les bienvenues! N'hésitez pas à:
1. Ouvrir une issue pour signaler un bug ou proposer une amélioration
2. Soumettre une pull request avec vos modifications
3. Améliorer la documentation

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.