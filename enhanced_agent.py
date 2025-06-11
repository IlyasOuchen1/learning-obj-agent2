import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
import tiktoken
from pinecone import Pinecone, ServerlessSpec

# Importer la configuration
try:
    from config import OPENAI_API_KEY, LLM_MODEL, AGENT_TEMPERATURE, VERBOSE_MODE, PINECONE_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
    AGENT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", 0.2))
    VERBOSE_MODE = os.environ.get("VERBOSE_MODE", "True").lower() == "true"
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

from components import (
    ObjectiveExtractor,
    ContentAnalyzer,
    BloomClassifier,
    ObjectiveFormatter,
    DifficultyEvaluator,
    LearningResourceRecommender,
    FeedbackGenerator
)

class DocumentProcessor:
    """Classe pour traiter et gérer les documents pédagogiques avec extraction d'objectifs"""
    
    def __init__(self, embedding_model="text-embedding-3-small", index_name="learn-obj"):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialiser Pinecone
        self._initialize_pinecone(index_name)
        
        # Text splitter optimisé pour les objectifs d'apprentissage
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Plus petit pour éviter les erreurs de tokens
            chunk_overlap=80,
            length_function=self._tiktoken_len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # LLM pour l'extraction d'objectifs depuis les documents
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=1000
        )
        
        # Prompt spécialisé pour l'extraction d'objectifs depuis les documents
        self.objective_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un expert en pédagogie spécialisé dans l'identification d'objectifs d'apprentissage dans les documents éducatifs.

Analysez le texte suivant et identifiez TOUS les objectifs d'apprentissage qu'il contient, qu'ils soient explicites ou implicites.

OBJECTIFS EXPLICITES - Recherchez des formulations comme :
- "À la fin de ce cours/module, l'étudiant sera capable de..."
- "Les objectifs de ce cours sont..."
- "L'apprenant devra être capable de..."
- "Compétences visées :"
- "Learning objectives:"
- "By the end of this course, students will be able to..."

OBJECTIFS IMPLICITES - Identifiez des phrases qui décrivent clairement ce que l'apprenant doit acquérir :
- "Ce cours enseigne..." → reformuler en objectif
- "Les étudiants apprendront..." → reformuler en objectif
- "Maîtriser les concepts de..." → reformuler en objectif

FORMAT DE RÉPONSE : Pour chaque objectif trouvé, utilisez ce format exact :
OBJECTIF: [reformulé sous forme "L'apprenant sera capable de..."]
TYPE: [explicite/implicite]
SOURCE: [citation exacte du texte original]
CONTEXTE: [document/section d'où provient l'objectif]

Si aucun objectif n'est trouvé, répondez : "AUCUN OBJECTIF IDENTIFIÉ"
"""),
            ("human", "TEXTE À ANALYSER:\n{text}")
        ])
    
    def _initialize_pinecone(self, index_name: str):
        """Initialise la connexion à Pinecone"""
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            existing_indexes = pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if index_name not in index_names:
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            self.index = pc.Index(index_name)
            self.index_name = index_name
            print(f"✅ Pinecone initialisé avec l'index: {index_name}")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de Pinecone: {e}")
            raise
    
    def _tiktoken_len(self, text: str) -> int:
        """Calcule la longueur d'un texte en tokens"""
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokens = tokenizer.encode(text, disallowed_special=())
            return len(tokens)
        except Exception:
            # Fallback simple
            return len(text.split())
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extrait le texte d'un fichier uploadé"""
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == "txt":
                loader = TextLoader(tmp_path, encoding='utf-8')
            elif file_extension in ["docx", "doc"]:
                loader = Docx2txtLoader(tmp_path)
            else:
                raise ValueError(f"Type de fichier non supporté: {file_extension}")
            
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            
            if not text.strip():
                raise ValueError("Le fichier ne contient pas de texte extractible")
                
            return text
            
        except Exception as e:
            print(f"Erreur lors de l'extraction du fichier {uploaded_file.name}: {e}")
            return f"Erreur d'extraction pour {uploaded_file.name}: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_documents(self, files: List, session_id: str) -> Tuple[List[str], List[str]]:
        """
        Traite une liste de fichiers et les stocke dans Pinecone
        
        Args:
            files: Liste des fichiers uploadés
            session_id: Identifiant de la session
            
        Returns:
            Tuple contenant les textes extraits et les noms des documents
        """
        document_texts = []
        document_names = []
        vectors_to_upsert = []
        
        for uploaded_file in files:
            try:
                print(f"📄 Traitement du fichier: {uploaded_file.name}")
                
                # Extraire le texte
                text = self.extract_text_from_file(uploaded_file)
                
                if "Erreur d'extraction" in text:
                    print(f"⚠️ Problème avec {uploaded_file.name}")
                    continue
                
                document_texts.append(text)
                document_names.append(uploaded_file.name)
                
                # Diviser en chunks
                chunks = self.text_splitter.split_text(text)
                print(f"📊 {len(chunks)} chunks créés pour {uploaded_file.name}")
                
                # Créer les vecteurs pour Pinecone
                for i, chunk in enumerate(chunks):
                    try:
                        embedding = self.embeddings.embed_query(chunk)
                        
                        vectors_to_upsert.append({
                            "id": f"{session_id}_{uploaded_file.name}_{i}",
                            "values": embedding,
                            "metadata": {
                                "text": chunk,
                                "source": uploaded_file.name,
                                "chunk": i,
                                "session_id": session_id,
                                "content_type": "educational_content",
                                "file_size": len(uploaded_file.getvalue())
                            }
                        })
                    except Exception as chunk_error:
                        print(f"⚠️ Erreur chunk {i} de {uploaded_file.name}: {chunk_error}")
                        continue
                        
            except Exception as file_error:
                print(f"❌ Erreur fichier {uploaded_file.name}: {file_error}")
                continue
        
        # Uploader vers Pinecone par batch
        if vectors_to_upsert:
            batch_size = 50
            total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    self.index.upsert(vectors=batch)
                    print(f"✅ Batch {batch_num}/{total_batches} uploadé ({len(batch)} vecteurs)")
                except Exception as batch_error:
                    print(f"❌ Erreur batch {batch_num}: {batch_error}")
                    continue
        
        print(f"🎉 Traitement terminé: {len(document_texts)} documents, {len(vectors_to_upsert)} chunks")
        return document_texts, document_names
    
    def search_relevant_content(self, query: str, session_id: str, top_k: int = 5) -> List[str]:
        """Recherche du contenu pertinent dans les documents de la session"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"session_id": {"$eq": session_id}}
            )
            
            relevant_content = []
            for match in results['matches']:
                content = f"Source: {match['metadata']['source']}\n{match['metadata']['text']}"
                relevant_content.append(content)
            
            return relevant_content
            
        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return []
    
    def extract_objectives_from_stored_documents(self, session_id: str) -> List[Dict]:
        """
        Extrait les objectifs d'apprentissage de tous les documents stockés dans Pinecone
        
        Args:
            session_id: ID de la session pour filtrer les documents
            
        Returns:
            Liste des objectifs trouvés avec leurs métadonnées
        """
        print(f"🔍 Recherche d'objectifs dans la session {session_id[:8]}...")
        
        # Mots-clés pour rechercher les sections contenant des objectifs
        objective_keywords = [
            "objectif apprentissage",
            "learning objective", 
            "compétence visée",
            "à la fin de ce cours",
            "l'étudiant sera capable",
            "students will be able",
            "learning outcome",
            "but pédagogique",
            "compétences développées",
            "skills acquired"
        ]
        
        all_objectives = []
        processed_chunks = set()  # Pour éviter les doublons
        
        # Rechercher pour chaque mot-clé
        for keyword in objective_keywords:
            try:
                query_embedding = self.embeddings.embed_query(keyword)
                
                results = self.index.query(
                    vector=query_embedding,
                    top_k=8,  # Résultats par mot-clé
                    include_metadata=True,
                    filter={"session_id": {"$eq": session_id}}
                )
                
                for match in results['matches']:
                    chunk_id = match['id']
                    
                    # Éviter de traiter le même chunk plusieurs fois
                    if chunk_id in processed_chunks:
                        continue
                    
                    processed_chunks.add(chunk_id)
                    
                    # Extraire les objectifs de ce chunk
                    chunk_text = match['metadata']['text']
                    source_doc = match['metadata']['source']
                    
                    objectives = self._extract_objectives_from_chunk(
                        chunk_text, 
                        source_doc,
                        match['score']
                    )
                    
                    all_objectives.extend(objectives)
                    
            except Exception as e:
                print(f"⚠️ Erreur recherche '{keyword}': {e}")
                continue
        
        print(f"📊 {len(processed_chunks)} chunks analysés, {len(all_objectives)} objectifs bruts trouvés")
        
        # Nettoyer et dédupliquer les objectifs
        unique_objectives = self._deduplicate_objectives(all_objectives)
        print(f"🎯 {len(unique_objectives)} objectifs uniques identifiés")
        
        return unique_objectives
    
    def _extract_objectives_from_chunk(self, text: str, source_doc: str, relevance_score: float) -> List[Dict]:
        """Extrait les objectifs d'un chunk de texte avec gestion d'erreur robuste"""
        try:
            # Vérifier la longueur du texte
            if len(text.strip()) < 20:
                return []
            
            # Utiliser le LLM pour extraire les objectifs
            chain = self.objective_extraction_prompt | self.llm
            result = chain.invoke({"text": text})
            response = result.content
            
            # Parser la réponse
            objectives = []
            
            if "AUCUN OBJECTIF IDENTIFIÉ" in response:
                return objectives
            
            # Pattern pour extraire les objectifs structurés
            pattern = r"OBJECTIF:\s*(.*?)\nTYPE:\s*(.*?)\nSOURCE:\s*(.*?)\nCONTEXTE:\s*(.*?)(?=\nOBJECTIF:|$)"
            matches = re.findall(pattern, response, re.DOTALL)
            
            for obj_text, obj_type, obj_source, obj_context in matches:
                # Nettoyer les données extraites
                clean_objective = obj_text.strip()
                clean_type = obj_type.strip().lower()
                clean_source = obj_source.strip()
                clean_context = obj_context.strip()
                
                # Valider que l'objectif est valide
                if len(clean_objective) < 10:  # Trop court
                    continue
                
                objective = {
                    "objective": clean_objective,
                    "type": clean_type,
                    "source_text": clean_source,
                    "context": clean_context,
                    "document_source": source_doc,      # Clé principale
                    "source_document": source_doc,      # Clé alternative
                    "source": source_doc,               # Clé de fallback
                    "relevance_score": float(relevance_score),
                    "extraction_method": "document_analysis",
                    "chunk_text": text[:200] + "..." if len(text) > 200 else text
                }
                objectives.append(objective)
            
            return objectives
            
        except Exception as e:
            print(f"⚠️ Erreur extraction objectifs: {e}")
            return []
    
    def _deduplicate_objectives(self, objectives: List[Dict]) -> List[Dict]:
        """Supprime les objectifs en double basés sur leur similitude"""
        if not objectives:
            return []
        
        unique_objectives = {}
        
        for obj in objectives:
            try:
                # Utiliser les premiers mots de l'objectif comme clé pour la déduplication
                objective_text = obj.get("objective", "")
                if len(objective_text) < 10:
                    continue
                    
                # Créer une clé basée sur les mots significatifs
                words = objective_text.lower().split()
                significant_words = [w for w in words if len(w) > 3][:8]  # 8 premiers mots significatifs
                key_words = " ".join(significant_words)
                
                # Si cet objectif (ou un très similaire) n'existe pas encore
                if key_words not in unique_objectives:
                    unique_objectives[key_words] = obj
                else:
                    # Garder celui avec le meilleur score de pertinence
                    current_score = obj.get("relevance_score", 0.0)
                    existing_score = unique_objectives[key_words].get("relevance_score", 0.0)
                    
                    if current_score > existing_score:
                        unique_objectives[key_words] = obj
                        
            except Exception as e:
                print(f"⚠️ Erreur déduplication: {e}")
                continue
        
        return list(unique_objectives.values())
    
    def get_all_document_content(self, session_id: str) -> List[Dict]:
        """Récupère tout le contenu des documents d'une session"""
        try:
            # Utiliser une requête large pour récupérer tous les chunks
            dummy_embedding = self.embeddings.embed_query("contenu document")
            
            results = self.index.query(
                vector=dummy_embedding,
                top_k=1000,  # Récupérer beaucoup de chunks
                include_metadata=True,
                filter={"session_id": {"$eq": session_id}}
            )
            
            # Organiser par document source
            documents_content = {}
            for match in results['matches']:
                source = match['metadata']['source']
                if source not in documents_content:
                    documents_content[source] = []
                
                documents_content[source].append({
                    "text": match['metadata']['text'],
                    "chunk": match['metadata']['chunk'],
                    "score": match['score']
                })
            
            # Reconstituer le texte complet de chaque document
            full_documents = []
            for source, chunks in documents_content.items():
                # Trier par numéro de chunk
                sorted_chunks = sorted(chunks, key=lambda x: x['chunk'])
                full_text = "\n".join([chunk['text'] for chunk in sorted_chunks])
                
                full_documents.append({
                    "source": source,
                    "full_text": full_text,
                    "num_chunks": len(chunks)
                })
            
            return full_documents
            
        except Exception as e:
            print(f"❌ Erreur récupération contenu: {e}")
            return []
    
    def clear_session_data(self, session_id: str):
        """Supprime tous les documents associés à une session"""
        try:
            self.index.delete(filter={"session_id": {"$eq": session_id}})
            print(f"🗑️ Données de session {session_id[:8]} supprimées")
        except Exception as e:
            print(f"❌ Erreur suppression: {e}")

class EnhancedLearningObjectiveAgent:
    """Agent principal qui coordonne l'analyse des objectifs avec la gestion de documents"""
    
    def __init__(self, api_key=None, model=None, temperature=None, verbose=None):
        # Configuration de l'API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        else:
            raise ValueError("Clé API OpenAI requise")
            
        # Paramètres du modèle
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else AGENT_TEMPERATURE
        self.verbose = verbose if verbose is not None else VERBOSE_MODE
        
        # Initialisation du modèle LLM
        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model)
            
        # Initialisation des composants d'analyse
        self.extractor = ObjectiveExtractor(self.llm)
        self.analyzer = ContentAnalyzer(self.llm)
        self.classifier = BloomClassifier(self.llm)
        self.formatter = ObjectiveFormatter(self.llm)
        self.evaluator = DifficultyEvaluator(self.llm)
        self.recommender = LearningResourceRecommender(self.llm)
        self.feedback_generator = FeedbackGenerator(self.llm)
        
        # Processeur de documents
        self.doc_processor = DocumentProcessor()
        
        # Session tracking
        self.current_session_id = None
        self.processed_documents = []
        self.extracted_document_objectives = []  # Objectifs extraits des documents
        
        # Création des outils avec gestion de documents
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
            ),
            Tool(
                name="search_documents",
                func=self.search_documents,
                description="Recherche des informations dans les documents uploadés"
            ),
            Tool(
                name="extract_document_objectives",
                func=self.extract_document_objectives,
                description="Extrait les objectifs d'apprentissage depuis les documents stockés"
            )
        ]
        
        # Création du prompt de l'agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Vous êtes un agent d'intelligence artificielle spécialisé dans l'analyse et l'amélioration des objectifs d'apprentissage selon la taxonomie de Bloom. 

Vous avez accès aux outils suivants pour vous aider dans votre analyse:
- extract_objectives: Pour extraire les objectifs d'apprentissage d'un texte
- analyze_content: Pour analyser le contenu pédagogique
- classify_objectives: Pour classifier selon la taxonomie de Bloom
- format_objectives: Pour reformuler les objectifs
- evaluate_difficulty: Pour évaluer la difficulté
- recommend_resources: Pour recommander des ressources
- generate_feedback: Pour générer du feedback
- search_documents: Pour rechercher dans les documents uploadés
- extract_document_objectives: Pour extraire les objectifs depuis les documents stockés

IMPORTANT: Quand des documents ont été uploadés, utilisez TOUJOURS l'outil extract_document_objectives pour identifier les objectifs d'apprentissage qui se trouvent déjà dans ces documents.

Pour chaque demande, suivez ces étapes:
1. Si des documents sont disponibles, extrayez d'abord les objectifs qu'ils contiennent
2. Recherchez des informations pertinentes supplémentaires dans les documents
3. Extrayez et analysez les objectifs d'apprentissage du contenu fourni
4. Combinez tous les objectifs trouvés (documents + contenu)
5. Classifiez-les selon la taxonomie de Bloom
6. Reformulez-les pour les améliorer
7. Évaluez leur difficulté
8. Recommandez des ressources appropriées
9. Générez un feedback constructif

Utilisez une approche étape par étape et expliquez clairement votre raisonnement.
            """),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Création de l'agent avec OpenAI Tools
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=self.verbose,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        print("✅ Agent d'objectifs d'apprentissage initialisé avec succès")
    
    def process_uploaded_files(self, files: List, session_id: str = None) -> str:
        """
        Traite les fichiers uploadés et les stocke pour utilisation ultérieure
        
        Args:
            files: Liste des fichiers uploadés
            session_id: Identifiant de session (généré automatiquement si non fourni)
            
        Returns:
            Résumé du traitement
        """
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        self.current_session_id = session_id
        
        try:
            print(f"🚀 Début du traitement - Session: {session_id[:8]}")
            
            document_texts, document_names = self.doc_processor.process_documents(files, session_id)
            
            self.processed_documents = []
            for i, (text, name) in enumerate(zip(document_texts, document_names)):
                self.processed_documents.append({
                    "name": name,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "full_text": text,
                    "word_count": len(text.split()),
                    "char_count": len(text)
                })
            
            # Extraire automatiquement les objectifs des documents uploadés
            print("🔍 Extraction des objectifs depuis les documents...")
            self.extracted_document_objectives = self.doc_processor.extract_objectives_from_stored_documents(session_id)
            
            result_message = f"✅ {len(files)} fichier(s) traité(s) avec succès."
            if self.extracted_document_objectives:
                result_message += f" {len(self.extracted_document_objectives)} objectif(s) d'apprentissage identifié(s) dans les documents."
            else:
                result_message += " Aucun objectif explicite trouvé, mais le contenu enrichira l'analyse."
            
            print(f"🎉 {result_message}")
            return result_message
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du traitement des fichiers: {str(e)}"
            print(error_msg)
            return error_msg
    
    def extract_document_objectives(self, query: str = "") -> str:
        """Extrait les objectifs d'apprentissage depuis les documents stockés"""
        if not self.current_session_id:
            return "Aucun document n'a été uploadé pour cette session."
        
        if not hasattr(self, 'extracted_document_objectives') or not self.extracted_document_objectives:
            # Extraire les objectifs si pas encore fait
            print("🔄 Extraction des objectifs en cours...")
            self.extracted_document_objectives = self.doc_processor.extract_objectives_from_stored_documents(
                self.current_session_id
            )
        
        if not self.extracted_document_objectives:
            return "Aucun objectif d'apprentissage n'a été trouvé dans les documents uploadés."
        
        # Formater les objectifs pour l'affichage
        result = f"📋 Objectifs d'apprentissage extraits des documents ({len(self.extracted_document_objectives)}):\n\n"
        
        for i, obj in enumerate(self.extracted_document_objectives, 1):
            source_doc = obj.get('document_source', obj.get('source_document', obj.get('source', 'Document inconnu')))
            obj_type = obj.get('type', 'non classifié')
            source_text = obj.get('source_text', 'Texte source non disponible')
            
            result += f"{i}. {obj['objective']}\n"
            result += f"   Type: {obj_type}\n"
            result += f"   Source: {source_doc}\n"
            result += f"   Extrait de: \"{source_text[:100]}...\"\n\n"
        
        return result
    
    def search_documents(self, query: str) -> str:
        """Recherche des informations dans les documents uploadés"""
        if not self.current_session_id:
            return "Aucun document n'a été uploadé pour cette session."
        
        relevant_content = self.doc_processor.search_relevant_content(
            query, self.current_session_id, top_k=5
        )
        
        if not relevant_content:
            return "Aucun contenu pertinent trouvé dans les documents uploadés."
        
        return "\n\n---\n\n".join(relevant_content)
    
    def extract_objectives(self, text: str) -> str:
        """Extrait les objectifs d'apprentissage d'un texte"""
        try:
            result = self.extractor.extract(text)
            return "\n".join(result)
        except Exception as e:
            print(f"⚠️ Erreur extraction objectifs: {e}")
            return "Erreur lors de l'extraction des objectifs"
    
    def analyze_content(self, content: str) -> str:
        """Analyse le contenu pédagogique"""
        try:
            result = self.analyzer.analyze(content)
            return result["analysis"]
        except Exception as e:
            print(f"⚠️ Erreur analyse contenu: {e}")
            return "Erreur lors de l'analyse du contenu"
    
    def classify_objectives(self, objectives: str) -> str:
        """Classifie les objectifs selon la taxonomie de Bloom"""
        try:
            objectives_list = [obj.strip() for obj in objectives.split("\n") if obj.strip()]
            result = self.classifier.classify(objectives_list)
            return result["classification"]
        except Exception as e:
            print(f"⚠️ Erreur classification: {e}")
            return "Erreur lors de la classification"
    
    def format_objectives(self, objectives: str) -> str:
        """Reformule et améliore les objectifs d'apprentissage"""
        try:
            objectives_list = [obj.strip() for obj in objectives.split("\n") if obj.strip()]
            result = self.formatter.format(objectives_list)
            return result["formatted_objectives"]
        except Exception as e:
            print(f"⚠️ Erreur formatage: {e}")
            return "Erreur lors de la reformulation"
    
    def evaluate_difficulty(self, objectives: str) -> str:
        """Évalue la difficulté des objectifs d'apprentissage"""
        try:
            objectives_list = [obj.strip() for obj in objectives.split("\n") if obj.strip()]
            result = self.evaluator.evaluate(objectives_list)
            return result["difficulty_evaluation"]
        except Exception as e:
            print(f"⚠️ Erreur évaluation difficulté: {e}")
            return "Erreur lors de l'évaluation de difficulté"
    
    def recommend_resources(self, objectives_with_levels: str) -> str:
        """Recommande des ressources d'apprentissage"""
        try:
            objectives_list = []
            bloom_levels = {}
            
            pattern = r"Objectif: (.*?)\nNiveau Bloom: (.*?)(?:\n|$)"
            matches = re.findall(pattern, objectives_with_levels, re.DOTALL)
            
            for obj, level in matches:
                objectives_list.append(obj.strip())
                bloom_levels[obj.strip()] = level.strip()
            
            result = self.recommender.recommend(objectives_list, bloom_levels)
            return result["recommendations"]
        except Exception as e:
            print(f"⚠️ Erreur recommandations: {e}")
            return "Erreur lors des recommandations"
    
    def generate_feedback(self, objectives_with_classifications: str) -> str:
        """Génère du feedback sur les objectifs d'apprentissage"""
        try:
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
        except Exception as e:
            print(f"⚠️ Erreur feedback: {e}")
            return "Erreur lors de la génération du feedback"
    
    def process_content_with_documents(self, content: str) -> Dict:
        """
        Traite le contenu pédagogique en enrichissant avec les documents uploadés
        
        Args:
            content: Le contenu pédagogique à analyser
            
        Returns:
            Les résultats de l'analyse enrichie
        """
        print("🚀 Début de l'analyse complète...")
        
        # Enrichir le contenu avec les informations des documents uploadés
        enriched_content = content
        
        if self.current_session_id and self.processed_documents:
            print("📄 Enrichissement avec les documents...")
            
            # Rechercher du contenu pertinent sur les objectifs d'apprentissage
            relevant_content = self.doc_processor.search_relevant_content(
                "objectifs d'apprentissage compétences", 
                self.current_session_id, 
                top_k=3
            )
            
            if relevant_content:
                enriched_content += "\n\n## Contenu extrait des documents uploadés:\n\n"
                enriched_content += "\n\n".join(relevant_content)
            
            # Ajouter les objectifs extraits des documents
            if hasattr(self, 'extracted_document_objectives') and self.extracted_document_objectives:
                enriched_content += "\n\n## Objectifs d'apprentissage trouvés dans les documents uploadés:\n\n"
                for obj in self.extracted_document_objectives:
                    source_doc = obj.get('document_source', obj.get('source_document', obj.get('source', 'Document inconnu')))
                    obj_type = obj.get('type', 'non classifié')
                    enriched_content += f"- {obj['objective']} (Source: {source_doc}, Type: {obj_type})\n"
        
        # Traitement direct pour obtenir les résultats structurés
        try:
            print("🎯 Extraction des objectifs du contenu...")
            # Extraire les objectifs du contenu fourni
            content_objectives = self.extractor.extract(enriched_content)
            
            # Combiner avec les objectifs extraits des documents
            all_objectives = content_objectives.copy()
            document_objectives_info = []
            
            if hasattr(self, 'extracted_document_objectives') and self.extracted_document_objectives:
                print(f"📄 Intégration de {len(self.extracted_document_objectives)} objectifs des documents...")
                
                for doc_obj in self.extracted_document_objectives:
                    # Ajouter à la liste globale
                    all_objectives.append(doc_obj['objective'])
                    
                    # Préparer les informations détaillées
                    source_doc = doc_obj.get('document_source', doc_obj.get('source_document', doc_obj.get('source', 'Document inconnu')))
                    
                    document_objectives_info.append({
                        "objective": doc_obj.get('objective', 'Objectif non spécifié'),
                        "type": doc_obj.get('type', 'non classifié'),
                        "source_document": source_doc,
                        "document_source": source_doc,  # Clé alternative
                        "source": source_doc,           # Clé de fallback
                        "source_text": doc_obj.get('source_text', 'Texte non disponible'),
                        "relevance_score": doc_obj.get('relevance_score', 0.0),
                        "context": doc_obj.get('context', 'Contexte non disponible')
                    })
            
            print(f"📊 Total: {len(all_objectives)} objectifs à analyser")
            
            # Analyser le contenu enrichi
            print("🔍 Analyse du contenu...")
            content_analysis = self.analyzer.analyze(enriched_content)
            
            # Classifier tous les objectifs
            print("🏷️ Classification selon Bloom...")
            classification_result = self.classifier.classify(all_objectives)
            
            print("✨ Reformulation des objectifs...")
            formatted_objectives = self.formatter.format(all_objectives)
            
            print("⚖️ Évaluation de la difficulté...")
            difficulty_evaluation = self.evaluator.evaluate(all_objectives)
            
            # Extraction des niveaux de Bloom
            print("🌸 Extraction des niveaux de Bloom...")
            bloom_levels = {}
            for obj in all_objectives:
                pattern = re.compile(rf"{re.escape(obj)}.*?Niveau de Bloom: ([a-zéè ]+)", re.DOTALL | re.IGNORECASE)
                match = pattern.search(classification_result["classification"])
                if match:
                    bloom_levels[obj] = match.group(1).strip()
                else:
                    bloom_levels[obj] = "non classifié"
            
            print("📚 Génération des recommandations...")
            recommendations = self.recommender.recommend(all_objectives, bloom_levels)
            
            # Extraction des classifications pour le feedback
            classifications = {}
            pattern = r"Objectif: (.*?)\n(.*?)(?=Objectif:|$)"
            matches = re.findall(pattern, classification_result["classification"], re.DOTALL)
            for obj, classification in matches:
                obj = obj.strip()
                classifications[obj] = classification.strip()
            
            print("💡 Génération du feedback...")
            feedback = self.feedback_generator.generate_feedback(all_objectives, classifications)
            
            # Compilation des résultats
            results = {
                "objectives": all_objectives,
                "content_objectives": content_objectives,
                "document_objectives": document_objectives_info,
                "content_analysis": content_analysis,
                "classification": classification_result,
                "formatted_objectives": formatted_objectives,
                "difficulty_evaluation": difficulty_evaluation,
                "recommendations": recommendations,
                "feedback": feedback,
                "stats": {
                    "total_objectives": len(all_objectives),
                    "content_objectives_count": len(content_objectives),
                    "document_objectives_count": len(document_objectives_info),
                    "processed_documents": len(self.processed_documents) if self.processed_documents else 0,
                    "session_id": self.current_session_id
                }
            }
            
            print("✅ Analyse terminée avec succès!")
            return results
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'analyse: {str(e)}"
            print(error_msg)
            
            return {
                "error": str(e),
                "objectives": [],
                "content_objectives": [],
                "document_objectives": [],
                "content_analysis": {"analysis": "Erreur lors de l'analyse"},
                "classification": {"classification": "Erreur lors de la classification"},
                "formatted_objectives": {"formatted_objectives": "Erreur lors de la reformulation"},
                "difficulty_evaluation": {"difficulty_evaluation": "Erreur lors de l'évaluation"},
                "recommendations": {"recommendations": "Erreur lors des recommandations"},
                "feedback": {"feedback": "Erreur lors de la génération du feedback"},
                "stats": {
                    "total_objectives": 0,
                    "content_objectives_count": 0,
                    "document_objectives_count": 0,
                    "processed_documents": 0,
                    "session_id": self.current_session_id
                }
            }
    
    def get_processed_documents_summary(self) -> List[Dict]:
        """Retourne un résumé des documents traités"""
        return self.processed_documents
    
    def get_extracted_document_objectives(self) -> List[Dict]:
        """Retourne les objectifs extraits des documents"""
        if hasattr(self, 'extracted_document_objectives'):
            return self.extracted_document_objectives
        return []
    
    def get_session_statistics(self) -> Dict:
        """Retourne les statistiques de la session"""
        return {
            "session_id": self.current_session_id,
            "processed_documents_count": len(self.processed_documents),
            "extracted_objectives_count": len(self.extracted_document_objectives) if hasattr(self, 'extracted_document_objectives') else 0,
            "total_words": sum([doc.get('word_count', 0) for doc in self.processed_documents]),
            "total_chars": sum([doc.get('char_count', 0) for doc in self.processed_documents])
        }
    
    def clear_session(self):
        """Efface les données de la session actuelle"""
        if self.current_session_id:
            print(f"🗑️ Nettoyage de la session {self.current_session_id[:8]}...")
            self.doc_processor.clear_session_data(self.current_session_id)
            self.current_session_id = None
            self.processed_documents = []
            self.extracted_document_objectives = []
            print("✅ Session nettoyée")
    
    def run(self, input_text: str) -> Dict:
        """Exécute l'agent avec le texte d'entrée"""
        try:
            result = self.agent_executor.invoke({"input": input_text})
            return result
        except Exception as e:
            print(f"❌ Erreur agent: {e}")
            return {"output": f"Erreur lors de l'exécution de l'agent: {str(e)}"}

# Fonction d'aide pour l'utilisation
def create_enhanced_agent(api_key=None, model=None):
    """Fonction d'aide pour créer un agent amélioré"""
    try:
        return EnhancedLearningObjectiveAgent(
            api_key=api_key,
            model=model
        )
    except Exception as e:
        print(f"❌ Erreur création agent: {e}")
        raise

# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("🧪 Test de l'agent d'objectifs d'apprentissage")
    
    try:
        # Créer l'agent
        agent = create_enhanced_agent()
        
        # Test avec du contenu simple
        content = """
        Ce cours vise à enseigner les bases de l'intelligence artificielle.
        À la fin de ce cours, les étudiants seront capables de:
        - Comprendre les concepts fondamentaux de l'IA
        - Implémenter des algorithmes d'apprentissage automatique simples
        - Analyser les performances des modèles
        """
        
        print("📝 Test d'analyse de contenu...")
        results = agent.process_content_with_documents(content)
        
        print("\n=== RÉSULTATS DE TEST ===")
        print(f"✅ Objectifs trouvés: {results['stats']['total_objectives']}")
        print(f"📊 Contenu: {results['stats']['content_objectives_count']}")
        print(f"📄 Documents: {results['stats']['document_objectives_count']}")
        
        if results.get("error"):
            print(f"❌ Erreur: {results['error']}")
        else:
            print("✅ Test réussi!")
            
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
    
    print("🎉 Fin des tests")