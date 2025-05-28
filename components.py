from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Optional
import re
from bloom_taxonomy import BloomTaxonomy

class ObjectiveExtractor:
    """Classe pour extraire des objectifs d'apprentissage d'un texte"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en pédagogie spécialisé dans l'extraction d'objectifs d'apprentissage.
            Analysez le texte fourni et extrayez tous les objectifs d'apprentissage explicites ou implicites.
            Un objectif d'apprentissage commence généralement par un verbe d'action et décrit ce que l'apprenant doit être capable de faire.
            Formatez chaque objectif comme une phrase complète commençant par "L'apprenant sera capable de" suivi d'un verbe d'action.
            Ne créez pas de nouveaux objectifs qui ne sont pas suggérés dans le texte."""),
            ("human", "{text}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def extract(self, text: str) -> List[str]:
        """Extrait les objectifs d'apprentissage d'un texte"""
        result = self.chain.invoke({"text": text})
        # Nettoyer et extraire les objectifs ligne par ligne
        objectives = [obj.strip() for obj in result["text"].split("\n") if obj.strip()]
        return objectives

class ContentAnalyzer:
    """Classe pour analyser le contenu pédagogique"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en analyse de contenu pédagogique.
            Analysez le contenu fourni et répondez aux questions suivantes:
            1. Quel est le sujet principal abordé?
            2. Quel est le niveau de complexité (débutant, intermédiaire, avancé)?
            3. Quels sont les concepts clés présentés?
            4. Quelles sont les compétences préalables nécessaires?
            5. Quels sont les points forts et les lacunes potentielles du contenu?
            Donnez une réponse détaillée pour chaque question."""),
            ("human", "{content}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, content: str) -> Dict:
        """Analyse le contenu pédagogique"""
        result = self.chain.invoke({"content": content})
        return {"analysis": result["text"]}

class BloomClassifier:
    """Classe pour classifier les objectifs selon la taxonomie de Bloom"""
    
    def __init__(self, llm):
        self.llm = llm
        self.bloom = BloomTaxonomy()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Vous êtes un expert de la taxonomie de Bloom. Voici les niveaux de la taxonomie avec leurs descriptions:
            
            {self._format_bloom_levels()}
            
            Pour chaque objectif d'apprentissage fourni, identifiez:
            1. Le verbe d'action principal
            2. Le niveau de Bloom correspondant
            3. Une justification de votre classification
            
            Formatez votre réponse pour chaque objectif comme:
            Objectif: [texte de l'objectif]
            Verbe principal: [verbe]
            Niveau de Bloom: [niveau]
            Justification: [votre justification]"""),
            ("human", "{objectives}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_bloom_levels(self) -> str:
        """Formate les niveaux de Bloom pour le prompt"""
        result = ""
        for level, data in BloomTaxonomy.LEVELS.items():
            result += f"{level.capitalize()}: {data['description']}\n"
            result += f"Verbes associés: {', '.join(data['verbs'])}\n\n"
        return result
    
    def classify(self, objectives: List[str]) -> Dict:
        """Classifie les objectifs selon la taxonomie de Bloom"""
        objectives_text = "\n".join(objectives)
        result = self.chain.invoke({"objectives": objectives_text})
        return {"classification": result["text"]}

class ObjectiveFormatter:
    """Classe pour reformuler et améliorer les objectifs d'apprentissage"""
    
    def __init__(self, llm):
        self.llm = llm
        self.bloom = BloomTaxonomy()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en formulation d'objectifs d'apprentissage.
            Pour chaque objectif fourni, reformulez-le pour qu'il soit:
            1. Spécifique - clairement défini et sans ambiguïté
            2. Mesurable - avec des critères d'évaluation clairs
            3. Atteignable - réaliste dans le contexte d'apprentissage
            4. Pertinent - lié au domaine d'étude
            5. Temporellement défini - avec une indication de quand il devrait être atteint
            
            Utilisez la structure "À la fin de [période], l'apprenant sera capable de [verbe d'action] [objet] [condition] [critère]."
            
            Exemple:
            Original: Comprendre les principes de la programmation orientée objet
            Reformulé: À la fin du module 3, l'apprenant sera capable d'expliquer les quatre principes fondamentaux de la programmation orientée objet et d'implémenter chacun d'eux dans un programme Java simple."""),
            ("human", "{objectives}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def format(self, objectives: List[str]) -> Dict:
        """Reformule et améliore les objectifs d'apprentissage"""
        objectives_text = "\n".join(objectives)
        result = self.chain.invoke({"objectives": objectives_text})
        return {"formatted_objectives": result["text"]}

class DifficultyEvaluator:
    """Classe pour évaluer la difficulté des objectifs d'apprentissage"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en évaluation de la difficulté des objectifs d'apprentissage.
            Pour chaque objectif fourni, évaluez sa difficulté sur une échelle de 1 à 5 où:
            1 = Très facile (connaissances de base)
            2 = Facile (application simple)
            3 = Modéré (analyse)
            4 = Difficile (évaluation complexe)
            5 = Très difficile (création originale)
            
            Pour chaque objectif, donnez:
            1. Le niveau de difficulté (1-5)
            2. Une justification de votre évaluation
            3. Une estimation du temps nécessaire pour l'atteindre
            4. Des conseils pour décomposer l'objectif en sous-objectifs plus faciles si la difficulté est ≥ 4"""),
            ("human", "{objectives}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def evaluate(self, objectives: List[str]) -> Dict:
        """Évalue la difficulté des objectifs d'apprentissage"""
        objectives_text = "\n".join(objectives)
        result = self.chain.invoke({"objectives": objectives_text})
        return {"difficulty_evaluation": result["text"]}

class LearningResourceRecommender:
    """Classe pour recommander des ressources d'apprentissage"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en ressources pédagogiques.
            Pour chaque objectif d'apprentissage, recommandez:
            1. 2-3 types d'activités d'apprentissage adaptées (exercices, projets, discussions, etc.)
            2. 2-3 méthodes d'évaluation appropriées
            3. 2-3 ressources génériques (types de documents, d'outils, etc.)
            
            Vos recommandations doivent être alignées avec le niveau de la taxonomie de Bloom de l'objectif.
            Soyez spécifique mais restez générique (ne mentionnez pas de produits ou services spécifiques)."""),
            ("human", "{objectives_with_levels}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def recommend(self, objectives: List[str], bloom_levels: Dict[str, str]) -> Dict:
        """Recommande des ressources d'apprentissage"""
        objectives_with_levels = []
        for obj in objectives:
            level = bloom_levels.get(obj, "non classifié")
            objectives_with_levels.append(f"Objectif: {obj}\nNiveau Bloom: {level}")
        
        objectives_text = "\n\n".join(objectives_with_levels)
        result = self.chain.invoke({"objectives_with_levels": objectives_text})
        return {"recommendations": result["text"]}

class FeedbackGenerator:
    """Classe pour générer du feedback sur les objectifs d'apprentissage"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en conception pédagogique spécialisé dans l'évaluation des objectifs d'apprentissage.
            Analysez les objectifs fournis et générez un feedback constructif sur:
            1. La couverture des différents niveaux de la taxonomie de Bloom
            2. L'équilibre entre les objectifs de bas niveau et de haut niveau
            3. La clarté et la précision des formulations
            4. La mesurabilité des objectifs
            5. Les lacunes potentielles ou les domaines non couverts
            
            Donnez des suggestions d'amélioration spécifiques et justifiées."""),
            ("human", "{objectives_with_classifications}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_feedback(self, objectives: List[str], classifications: Dict) -> Dict:
        """Génère du feedback sur les objectifs d'apprentissage"""
        objectives_text = "\n\n".join([f"Objectif: {obj}\n{classifications.get(obj, 'Non classifié')}" for obj in objectives])
        result = self.chain.invoke({"objectives_with_classifications": objectives_text})
        return {"feedback": result["text"]}