from typing import List, Optional

class BloomTaxonomy:
    """Classe qui définit la taxonomie de Bloom avec ses niveaux et verbes associés"""
    
    LEVELS = {
        "se souvenir": {
            "description": "Récupérer, reconnaître et rappeler des connaissances pertinentes de la mémoire à long terme",
            "verbs": ["définir", "décrire", "identifier", "étiqueter", "lister", "mémoriser", 
                      "nommer", "reconnaître", "réciter", "rappeler", "répéter"]
        },
        "comprendre": {
            "description": "Déterminer le sens des messages pédagogiques, qu'ils soient oraux, écrits ou graphiques",
            "verbs": ["clarifier", "classer", "comparer", "convertir", "défendre", "démontrer", 
                      "différencier", "discuter", "distinguer", "estimer", "expliquer", 
                      "généraliser", "donner des exemples", "illustrer", "inférer", 
                      "interpréter", "paraphraser", "prédire", "reformuler", "résumer", "traduire"]
        },
        "appliquer": {
            "description": "Exécuter ou utiliser une procédure dans une situation donnée",
            "verbs": ["appliquer", "calculer", "construire", "démontrer", "développer", 
                      "découvrir", "manipuler", "modifier", "opérer", "prédire", 
                      "préparer", "produire", "résoudre", "utiliser"]
        },
        "analyser": {
            "description": "Décomposer un matériel en ses parties constitutives et déterminer comment les parties sont liées entre elles et à une structure ou un objectif global",
            "verbs": ["analyser", "déconstruire", "attribuer", "différencier", "discriminer", 
                      "distinguer", "examiner", "expérimenter", "identifier", "illustrer", 
                      "inférer", "schématiser", "esquisser", "structurer", "catégoriser"]
        },
        "évaluer": {
            "description": "Porter des jugements basés sur des critères et des normes",
            "verbs": ["évaluer", "argumenter", "défendre", "juger", "sélectionner", 
                      "supporter", "valoriser", "critiquer", "pondérer", "justifier", 
                      "mesurer", "comparer", "conclure", "discriminer", "expliquer", "interpréter"]
        },
        "créer": {
            "description": "Assembler des éléments pour former un tout cohérent ou fonctionnel; réorganiser des éléments en un nouveau modèle ou une nouvelle structure",
            "verbs": ["assembler", "construire", "créer", "concevoir", "développer", 
                      "formuler", "écrire", "inventer", "fabriquer", "planifier", 
                      "produire", "proposer", "configurer", "synthétiser", "établir"]
        }
    }

    @classmethod
    def get_all_verbs(cls) -> List[str]:
        """Récupère tous les verbes de la taxonomie de Bloom"""
        all_verbs = []
        for level_data in cls.LEVELS.values():
            all_verbs.extend(level_data["verbs"])
        return all_verbs
    
    @classmethod
    def classify_verb(cls, verb: str) -> Optional[str]:
        """Classifie un verbe selon les niveaux de Bloom"""
        verb = verb.lower()
        for level, level_data in cls.LEVELS.items():
            if verb in level_data["verbs"]:
                return level
        return None
    
    @classmethod
    def get_level_description(cls, level: str) -> str:
        """Obtient la description d'un niveau de Bloom"""
        if level in cls.LEVELS:
            return cls.LEVELS[level]["description"]
        return "Niveau non reconnu dans la taxonomie de Bloom"
    
    @classmethod
    def get_alternative_verbs(cls, level: str) -> List[str]:
        """Obtient des verbes alternatifs pour un niveau donné"""
        if level in cls.LEVELS:
            return cls.LEVELS[level]["verbs"]
        return []