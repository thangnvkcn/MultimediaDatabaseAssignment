# import the necessary packages
import numpy as np
import csv
from math import *
class Searcher:
	def __init__(self, indexPath):
		# store our index path
	    self.indexPath = indexPath

	def search(self, queryFeatures, limit = 10):
		results = {}
		with open(self.indexPath) as f:
			reader = csv.reader(f)
			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.euclidean_distance(features, queryFeatures)
				results[row[0]] = d
			f.close()
		results = sorted([(v, k) for (k, v) in results.items()])
		return results[:limit]

	def chi2_distance(self, histA, histB, eps=1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
						  for (a, b) in zip(histA, histB)])
		return d

	def euclidean_distance(self,x,y):
		return sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

	def square_rooted(self,x):
		return round(sqrt(sum([a*a for a in x])),3)
	def cosin_similarity(self,x,y):
		numerator = sum(a*b for a,b in zip(x,y))
		denominator = self.square_rooted(x)*self.square_rooted(y)
		return round(numerator/float(denominator),3)

	def manhattan_distance(self,x,y):
		return sum(abs(a-b) for a,b in zip(x,y))