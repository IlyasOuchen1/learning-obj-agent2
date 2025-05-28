from flask import Flask, render_template, request, jsonify
import os
from agent import LearningObjectiveAgent

app = Flask(__name__)

# Initialisation de l'agent (utilisez votre clé API)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
agent = LearningObjectiveAgent(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    content = request.form['content']
    
    # Traitement du contenu avec l'agent
    results = agent.process_content(content)
    
    return jsonify({
        'objectives': results['objectives'],
        'content_analysis': results['content_analysis']['analysis'],
        'classification': results['classification']['classification'],
        'formatted_objectives': results['formatted_objectives']['formatted_objectives'],
        'difficulty_evaluation': results['difficulty_evaluation']['difficulty_evaluation'],
        'recommendations': results['recommendations']['recommendations'],
        'feedback': results['feedback']['feedback']
    })

if __name__ == '__main__':
    # Assurez-vous de créer le dossier templates et d'y placer index.html
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)