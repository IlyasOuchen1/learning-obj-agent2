import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# Configuration de l'agent
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", 0.2))
VERBOSE_MODE = os.getenv("VERBOSE_MODE", "True").lower() == "true"

# Configuration Flask
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG = int(os.getenv("FLASK_DEBUG", 1))
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

# Autres configurations
PROJECT_NAME = os.getenv("PROJECT_NAME", "Learning Objective Agent")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "./outputs")

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Vérifier que la clé API OpenAI est définie
if not OPENAI_API_KEY:
    print("ATTENTION: La clé API OpenAI n'est pas définie dans le fichier .env")
    print("Définissez OPENAI_API_KEY dans le fichier .env ou en tant que variable d'environnement")