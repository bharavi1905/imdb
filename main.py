import numpy as np
import pandas as pd
import flask

import json, requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ShowSimilarity(object):
	def __init__(self):
		self.__load_data()

	def __load_data(self):
		#self.__show_data = pickle.load(open('shows_data_filtered.pickle', 'rb'))
		self.__show_data = pd.read_csv('shows.csv')
        	self.__show_data = self.__show_data.loc[:6001]
		print('Loaded book data.')
		cv = CountVectorizer()
		count_matrix = cv.fit_transform(self.__show_data['soup'])
		self.__cos_sim = cosine_similarity(count_matrix, count_matrix)
		self.__title_to_idx = pd.Series(self.__show_data.index, index=self.__show_data['title']) # title -> idx mapping
		print('Loaded cosine similarity matrix.')

	def search(self, query):
		return self.__show_data.loc[self.__show_data['title'].str.contains(query, case=False)]

	def recommend(self, title):
		if title not in self.__title_to_idx:
			print('Title not found in index mapping.')
			return None

		# Get the index of the passed in book's title.
		show_idx = self.__title_to_idx[title]

		# Get scores from the cosine similarity matrix for this index.
		scores = pd.Series(self.__cos_sim[show_idx]).sort_values(ascending=False)
		# Get the indices of the top 10 books (sub the first as it's the input book).
		indices = list(scores.iloc[:11].index)

		return self.__show_data.iloc[indices]

app = flask.Flask(__name__)
showsim = ShowSimilarity()

@app.route('/')
def index():
	return flask.render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
	search = flask.request.args.get('q')
	filtered_shows = list(showsim.search(search)['title'].values)
	return flask.jsonify(matching_results=filtered_shows)

@app.route('/recommend')
def recommend():
	# Dynamic page help:
	# https://stackoverflow.com/questions/40963401/flask-dynamic-data-update-without-reload-page/40964086
	
	searchText = flask.request.args.get('jsdata')

	output = dict()
	if searchText:
		#print(f'Search text: {searchText}')
		results = showsim.recommend(searchText)
		if results is not None:
			output = results[['title', 'year', 'language', 'genre', 'rating',\
			                  	'stars', 'creators', 'episode_season_duration', 'description', 'trailer', 'nameurls']].to_dict(orient='records')#results.title.values

	# TODO: Convert a fuller version to JSON rather than just an array (title, url, etc.) and render as a table instead.
	# https://stackoverflow.com/questions/48050769/pandas-dataframe-to-flask-template-as-a-json
	#print(output)
	return flask.render_template('results.html', recommendations=output)


if __name__ == '__main__':
    app.run()

#global data
# def create_similarity():
#     data = pd.read_csv('shows_dlg.csv')
#     data = data.loc[:6001]
#     print('Loaded book data')
#     cv = CountVectorizer(dtype=np.float32)
#     count_matrix = cv.fit_transform(data['soup'])
#     print('count')
#     # create a similarity score matrix
#     similarity = cosine_similarity(count_matrix).astype('float16')
#     print('sim')
#     title_to_idx = pd.Series(data.index, index=data['title'])
#     print('title')
#     return similarity, title_to_idx


# def search_title(query):
#     data = pd.read_csv('shows_dlg.csv')
#     return data.loc[data['title'].str.contains(query, case=False)]


# def recommend_index(title):
#     similarity, title_to_idx = create_similarity()
#     if title not in title_to_idx:
#         print('Title not found in index mapping')
#         return None

#     show_idx = title_to_idx[title]
#     scores = pd.Series(similarity[show_idx]).sort_values(ascending=False)
#     indices = list(scores.iloc[:11].index)
#     return indices


# app = flask.Flask(__name__)

# @app.route('/')
# def index():
#     return flask.render_template('index.html')

# @app.route('/autocomplete', methods=['GET'])
# def autocomplete():
#     search = flask.request.args.get('q')
#     filtered_shows = list(search_title(search)['title'].values)
#     return flask.jsonify(matching_results=filtered_shows)

# @app.route('/recommend', methods=['GET'])
# def recommend():
#     searchText = flask.request.args.get('jsdata')
#     results = []
#     if searchText:
#         indexes = recommend_index(searchText)
#         for idx in indexes:
#             req = requests.get('https://imdbdb.herokuapp.com/data/{}'.format(idx))
#             json_data = json.loads(req.content)
#             results.append(json_data)

#         print(results)
#         return flask.render_template('results.html', recommendations=results)

# if __name__ == '__main__':
#     app.run()

    
    
    
