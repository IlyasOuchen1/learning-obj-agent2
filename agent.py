from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional
import re
import os

# Importer la configuration
try:
    from config import OPENAI_API_KEY, LLM_MODEL, AGENT_TEMPERATURE, VERBOSE_MODE
except ImportError:
    # Fallback si config.py n'est pas disponible
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    AGENT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", 0.2))
    VERBOSE_MODE = os.environ.get("VERBOSE_MODE", "True").lower() == "true"

from components import (
    ObjectiveExtractor,
    ContentAnalyzer,
    BloomClassifier,
    ObjectiveFormatter,
    DifficultyEvaluator,
    LearningResourceRecommender,
    FeedbackGenerator
)

class LearningObjectiveAgent:
    """Agent principal qui coordonne tous les composants"""
    
    def __init__(self, api_key=None, model=None, temperature=None, verbose=None):
        # Configuration de l'API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
        # Paramètres du modèle
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else AGENT_TEMPERATURE
        self.verbose = verbose if verbose is not None else VERBOSE_MODE
        
        # Initialisation du modèle LLM
        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model)
            
        # Initialisation des composants
        self.extractor = ObjectiveExtractor(self.llm)
        self.analyzer = ContentAnalyzer(self.llm)
        self.classifier = BloomClassifier(self.llm)
        self.formatter = ObjectiveFormatter(self.llm)
        self.evaluator = DifficultyEvaluator(self.llm)
        self.recommender = LearningResourceRecommender(self.llm)
        self.feedback_generator = FeedbackGenerator(self.llm)
        
        # Mémoire pour stocker les résultats
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Création des outils
        self.tools = [
            Tool(
                name="extract_objectives",
                func=self.extract_objectives,
                description="Extrait les objectifs d'apprentissage d'un texte"
            ),
            Tool(
                name="analyze_content",
                func=self.analyze_content,
                description="Analyse le contenu pédagogique"
            ),
            Tool(
                name="classify_objectives",
                func=self.classify_objectives,
                description="Classifie les objectifs selon la taxonomie de Bloom"
            ),
            Tool(
                name="format_objectives",
                func=self.format_objectives,
                description="Reformule et améliore les objectifs d'apprentissage"
            ),
            Tool(
                name="evaluate_difficulty",
                func=self.evaluate_difficulty,
                description="Évalue la difficulté des objectifs d'apprentissage"
            ),
            Tool(
                name="recommend_resources",
                func=self.recommend_resources,
                description="Recommande des ressources d'apprentissage"
            ),
            Tool(
                name="generate_feedback",
                func=self.generate_feedback,
                description="Génère du feedback sur les objectifs d'apprentissage"
            )
        ]
        
        # Création du prompt de l'agent - CORRIGÉ POUR INCLURE LES VARIABLES MANQUANTES
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un agent d'intelligence artificielle spécialisé dans l'analyse et l'amélioration des objectifs d'apprentissage selon la taxonomie de Bloom. Votre but est d'aider les concepteurs pédagogiques à créer des objectifs d'apprentissage efficaces.

Voici les outils à votre disposition:
{tools}

Voici les noms des outils disponibles:
{tool_names}

Pour chaque demande, suivez ces étapes:
1. Extrayez et analysez les objectifs d'apprentissage
2. Classifiez-les selon la taxonomie de Bloom
3. Reformulez-les pour les améliorer
4. Évaluez leur difficulté
5. Recommandez des ressources appropriées
6. Générez un feedback constructif

Utilisez une approche étape par étape et expliquez clairement votre raisonnement.
            """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Création de l'agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=self.verbose)
    
    def extract_objectives(self, text: str) -> str:
        """Extrait les objectifs d'apprentissage d'un texte"""
        result = self.extractor.extract(text)
        return "\n".join(result)
    
    def analyze_content(self, content: str) -> str:
        """Analyse le contenu pédagogique"""
        result = self.analyzer.analyze(content)
        return result["analysis"]
    
    def classify_objectives(self, objectives: str) -> str:
        """Classifie les objectifs selon la taxonomie de Bloom"""
        objectives_list = objectives.split("\n")
        result = self.classifier.classify(objectives_list)
        return result["classification"]
    
    def format_objectives(self, objectives: str) -> str:
        """Reformule et améliore les objectifs d'apprentissage"""
        objectives_list = objectives.split("\n")
        result = self.formatter.format(objectives_list)
        return result["formatted_objectives"]
    
    def evaluate_difficulty(self, objectives: str) -> str:
        """Évalue la difficulté des objectifs d'apprentissage"""
        objectives_list = objectives.split("\n")
        result = self.evaluator.evaluate(objectives_list)
        return result["difficulty_evaluation"]
    
    def recommend_resources(self, objectives_with_levels: str) -> str:
        """Recommande des ressources d'apprentissage"""
        # Parsing du format "Objectif: X\nNiveau Bloom: Y"
        objectives_list = []
        bloom_levels = {}
        
        pattern = r"Objectif: (.*?)\nNiveau Bloom: (.*?)(?:\n|$)"
        matches = re.findall(pattern, objectives_with_levels, re.DOTALL)
        
        for obj, level in matches:
            objectives_list.append(obj.strip())
            bloom_levels[obj.strip()] = level.strip()
        
        result = self.recommender.recommend(objectives_list, bloom_levels)
        return result["recommendations"]
    
    def generate_feedback(self, objectives_with_classifications: str) -> str:
        """Génère du feedback sur les objectifs d'apprentissage"""
        # Parsing du format "Objectif: X\nClassification: Y"
        objectives_list = []
        classifications = {}
        
        pattern = r"Objectif: (.*?)\n(.*?)(?=\nObjectif:|$)"
        matches = re.findall(pattern, objectives_with_classifications, re.DOTALL)
        
        for obj, classification in matches:
            obj = obj.strip()
            objectives_list.append(obj)
            classifications[obj] = classification.strip()
        
        result = self.feedback_generator.generate_feedback(objectives_list, classifications)
        return result["feedback"]
    
    def run(self, input_text: str) -> Dict:
        """Exécute l'agent avec le texte d'entrée"""
        result = self.agent_executor.invoke({"input": input_text})
        return result
    
    def process_content(self, content: str) -> Dict:
        """Traite le contenu pédagogique sans utiliser l'agent (processus direct)"""
        # Extraction des objectifs
        objectives = self.extractor.extract(content)
        
        # Analyse du contenu
        content_analysis = self.analyzer.analyze(content)
        
        # Classification des objectifs
        classification_result = self.classifier.classify(objectives)
        
        # Formatage des objectifs
        formatted_objectives = self.formatter.format(objectives)
        
        # Évaluation de la difficulté
        difficulty_evaluation = self.evaluator.evaluate(objectives)
        
        # Extraction des niveaux de Bloom à partir de la classification
        bloom_levels = {}
        for obj in objectives:
            # On extrait le niveau de Bloom de la classification
            pattern = re.compile(rf"{re.escape(obj)}.*?Niveau de Bloom: ([a-zéè ]+)", re.DOTALL | re.IGNORECASE)
            match = pattern.search(classification_result["classification"])
            if match:
                bloom_levels[obj] = match.group(1).strip()
            else:
                bloom_levels[obj] = "non classifié"
        
        # Recommandation de ressources
        recommendations = self.recommender.recommend(objectives, bloom_levels)
        
        # Extraction des classifications détaillées pour le feedback
        classifications = {}
        pattern = r"Objectif: (.*?)\n(.*?)(?=Objectif:|$)"
        matches = re.findall(pattern, classification_result["classification"], re.DOTALL)
        for obj, classification in matches:
            obj = obj.strip()
            classifications[obj] = classification.strip()
        
        # Génération de feedback
        feedback = self.feedback_generator.generate_feedback(objectives, classifications)
        
        # Assemblage des résultats
        return {
            "objectives": objectives,
            "content_analysis": content_analysis,
            "classification": classification_result,
            "formatted_objectives": formatted_objectives,
            "difficulty_evaluation": difficulty_evaluation,
            "recommendations": recommendations,
            "feedback": feedback
        }