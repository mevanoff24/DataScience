from pyspark import SparkConf, SparkContext
import sys
from math import sqrt

# pseudoCode

# 1 - load txt files 
# 2 - split data into (userID, movieID, rating)
# 3 - self-join all movies and create all possible combinations of movie pairs
# 4 - remove all duplicate movie pairs
# 5 - combine movies
# 6 - computed similarities between movies 
# 7 - set thresholds and sort keys and only return movies that meet criteria
# 8 - make action, map movieID => movieName and display results  


# set config
conf = SparkConf().setMaster('local[*]').setAppName('MovieSimilarity')
sc = SparkContext(conf = conf)

def loadMovieNames():
	movieNames = {}
	with open('data/ml-100k/u.ITEM') as f:
		for line in f:
			fields = line.split('|')
			movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
	return movieNames


def makePairs((user, ratings)):
	(movie1, rating1) = ratings[0]
	(movie2, rating2) = ratings[1]
	return ( (movie1, movie2), (rating1, rating2) )


def filterDups((userID, ratings)):
	(movie1, rating1) = ratings[0]
	(movie2, rating2) = ratings[1]
	return movie1 < movie2


def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


# MAIN PROGRAM

nameDict = loadMovieNames()

data = sc.textFile('data/ml-100k/u.data')

# returns (userID, movieID, rating)
ratings = (data
		  .map(lambda x: x.split())
		  .map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))
		  )

# self-join for every possilbe combination of movies
# returns (userID, ((movieID, rating), (movieID, rating)))
# sample  (512, ((265, 4.0), (265, 4.0)))
joinedRatings = ratings.join(ratings)

# remove duplicate movies after self-join
uniqueJoinedRatings = joinedRatings.filter(filterDups)

# create all movies pairs
# returns ((movie1, movie2), (rating1, rating2))
moviePairs = uniqueJoinedRatings.map(makePairs)

# combine unique movie ratings
moviePairRatings = moviePairs.groupByKey()

# compute similarities between movies and cache RDD 
# returns ((movieID_1, movieID_2), (similarityScore, numberPairs))
# sample ((197, 1097), (0.9758729093599599, 7))
moviePairSimilarities = (moviePairRatings
						.mapValues(computeCosineSimilarity)
						.cache())

# Save results 
# moviePairSimilarities.sortByKey()
# moviePairSimilarities.saveAsTextFile("movie-sims")

if len(sys.argv) > 1:

	scoreThreshold = 0.90
	coOccurenceThreshold = 50

	movieID = int(sys.argv[1])

	# filter movies that contain movieID and use desired similarity and rating count thresholds
	filteredResults = (moviePairSimilarities
					  .filter(lambda ((pair, sim)): (pair[0] == movieID or pair[1] == movieID) \
        				  and sim[0] > scoreThreshold and sim[1] > coOccurenceThreshold))

	results = (filteredResults
			  .map(lambda (x, y): (y, x))
			  .sortByKey(ascending = False)
			  .take(10))

	print '\n'
	print "Top 10 similar movies for " + nameDict[movieID]

	for result in results:
		(sim, pair) = result
		similarMovieID = pair[0]

		if similarMovieID == movieID:
			similarMovieID = pair[1]
		print nameDict[similarMovieID] + "\tscore:" + str(sim[0]) + "\tstrength:" + str(sim[1])

	print '\n'



