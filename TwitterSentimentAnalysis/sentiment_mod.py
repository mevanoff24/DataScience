from __future__ import division
import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_clfs/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()



word_features5k_f = open("pickled_clfs/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



# feature_set_f = open("pickled_clfs/featuresets.pickle", "rb")
# feature_set = pickle.load(feature_set_f)
# feature_set_f.close()

# random.shuffle(feature_set)
# print(len(feature_set))

# X_test = feature_set[10000:]
# X_train = feature_set[:10000]

def open_file(filename):
    open_file = open('pickled_clfs/' + filename + '.pickle', 'rb')
    clf = pickle.load(open_file)
    open_file.close() 
    return clf

classifier = open_file('originalnaivebayes')

MNB_classifier = open_file('MultinomialNB')

BernoulliNB_classifier = open_file('BernoulliNB')

LogisticRegression_classifier = open_file('LogisticRegression')

LinearSVC_classifier = open_file('LinearSVC')

SGDC_classifier = open_file('SGDClassifier')


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)



def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)