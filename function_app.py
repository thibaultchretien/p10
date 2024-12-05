import azure.functions as func
from flask import Flask, jsonify, request
from azure.functions import WsgiMiddleware
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# Charger les fichiers CSV
current_dir = os.path.dirname(__file__)
articles = pd.read_csv(os.path.join(current_dir, 'articles_metadata.csv'))
clicks = pd.read_csv(os.path.join(current_dir, 'clicks_sample.csv'))

# Prétraitement des données des articles
articles['created_at_ts'] = pd.to_datetime(articles['created_at_ts'], unit='ms')
articles['article_age_days'] = (pd.Timestamp.now() - articles['created_at_ts']).dt.days

# Normalisation des caractéristiques continues
scaler = StandardScaler()
articles[['words_count', 'article_age_days']] = scaler.fit_transform(
    articles[['words_count', 'article_age_days']]
)

article_features = articles[['category_id', 'words_count', 'article_age_days']].copy()

# Fonction pour calculer la similarité cosinus
def get_cosine_similarity_for_article(article_id):
    if article_id not in articles['article_id'].values:
        return {"error": f"Article {article_id} non trouvé."}, 404

    article_index = articles[articles['article_id'] == article_id].index[0]
    similarity_matrix = cosine_similarity([article_features.iloc[article_index]], article_features)
    similarity_scores = similarity_matrix[0]
    similar_articles_indices = similarity_scores.argsort()[::-1]

    similar_articles = articles.iloc[similar_articles_indices]
    similar_articles['similarity_score'] = similarity_scores[similar_articles_indices]
    similar_articles = similar_articles[similar_articles['article_id'] != article_id]
    return similar_articles[['article_id', 'category_id', 'similarity_score']].head(5).to_dict(orient='records')

# Initialiser Flask
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    article_id = request.args.get('article_id')
    if not article_id:
        return jsonify({"error": "Veuillez fournir un article_id."}), 400
    try:
        article_id = int(article_id)
    except ValueError:
        return jsonify({"error": "L'article_id doit être un entier."}), 400

    recommendations = get_cosine_similarity_for_article(article_id)
    if isinstance(recommendations, dict):  # Si l'erreur est retournée sous forme de dictionnaire
        return jsonify(recommendations), 404
    return jsonify(recommendations)

# Fonction d'entrée pour Azure Functions
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    # Créer un middleware WSGI pour connecter Flask à Azure Functions
    return WsgiMiddleware(app.wsgi_app).handle(req, context)

