import numpy as np
import pandas as pd
import flask
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, requests

class ShowSimilarity(object):
	def __init__(self):
		self.__load_data()

	def __load_data(self):
		#self.__show_data = pickle.load(open('shows_data_filtered.pickle', 'rb'))
		self.__show_data = pd.read_csv('shows_compressed.csv')
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

		return indices      #, self.__show_data.iloc[indices]


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

@app.route('/recommend', methods=['GET'])
def recommend():

	searchText = flask.request.args.get('jsdata')
	results = []
	if searchText:
		indexes = showsim.recommend(searchText)
		for idx in indexes:
			req = requests.get('https://imdbdb.herokuapp.com/data/{}'.format(idx))
			json_data = json.loads(req.content)
			results.append(json_data)
			
	#print(results)
	return flask.render_template('results.html', recommendations=results)

if __name__ == '__main__':
    app.run()
