import numpy as np
import pandas as pd
import flask

import json, requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#global data
def create_similarity():
    data = pd.read_csv('shows_dlg.csv')
    print('Loaded book data')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['soup'])
    # create a similarity score matrix
    similarity = cosine_similarity(count_matrix, count_matrix)
    print('sim')
    title_to_idx = pd.Series(data.index, index=data['title'])
    print('title')
    return similarity, title_to_idx


def search_title(query):
    data = pd.read_csv('shows_dlg.csv')
    return data.loc[data['title'].str.contains(query, case=False)]


def recommend_index(title):
    similarity, title_to_idx = create_similarity()
    if title not in title_to_idx:
        print('Title not found in index mapping')
        return None

    show_idx = title_to_idx[title]
    scores = pd.Series(similarity[show_idx]).sort_values(ascending=False)
    indices = list(scores.iloc[:11].index)
    return indices


app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = flask.request.args.get('q')
    filtered_shows = list(search_title(search)['title'].values)
    return flask.jsonify(matching_results=filtered_shows)

@app.route('/recommend', methods=['GET'])
def recommend():
    searchText = flask.request.args.get('jsdata')
    results = []
    if searchText:
        indexes = recommend_index(searchText)
        for idx in indexes:
            req = requests.get('https://imdbdb.herokuapp.com/data/{}'.format(idx))
            json_data = json.loads(req.content)
            results.append(json_data)

        print(results)
        return flask.render_template('results.html', recommendations=results)

if __name__ == '__main__':
    app.run()

    
    
    
