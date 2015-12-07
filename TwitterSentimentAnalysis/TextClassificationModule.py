from __future__ import division
from statistics import mode
import pickle
import random

import nltk 
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC


class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # classify post based on the mode (most common) decision by classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # compute percent confidence of classifiers 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
# read in data    
short_pos = open("review_data/short_review_pos.txt","r").read().decode('utf-8')
short_neg = open("review_data/short_review_neg.txt","r").read().decode('utf-8')

all_words = []
documents = []

# fill words and documents from text files 
#  J is adject, R is adverb, and V is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_documents = open("pickled_clfs/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)

# use top 5000 words as features 
# word_features = list(all_words.keys())[:5000]
word_features = [w for (w, c) in all_words.most_common(5000)]

# pickle word features
save_word_features = open("pickled_clfs/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# create features 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_set = [(find_features(rev), category) for (rev, category) in documents]

# shuffle features
random.shuffle(feature_set)
print len(feature_set) 

# train / test sets
X_test = feature_set[10000:]
X_train = feature_set[:10000]


# NLTK NAIVE BAYES
classifier = nltk.NaiveBayesClassifier.train(X_train)
print nltk.classify.accuracy(classifier, X_test)
# classifier.show_most_informative_features(15)
save_classifier = open("pickled_clfs/originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

###############
# SKLEARN CLASSIFIERS 

# train desired classifier 
def train_classifier(clf_name, train, test, name):
    out_folder = 'pickled_clfs/'
    clf = SklearnClassifier(clf_name)
    clf.train(train)
    print nltk.classify.accuracy(clf, test)
    return clf

# pickle desired classifier after train so only have to run once
def pickle_classifier(clf_name, train, test, name):
    clf_title = train_classifier(clf_name, X_train, X_test, 'MultinomialNB')
    out_folder = 'pickled_clfs/'
    save_classifier = open(out_folder + name + '.pickle','wb')
    pickle.dump(clf_title, save_classifier)
    save_classifier.close()

pickle_classifier(MultinomialNB(), X_train, X_test, 'MultinomialNB')

pickle_classifier(BernoulliNB(), X_train, X_test, 'BernoulliNB')

pickle_classifier(LogisticRegression(), X_train, X_test, 'LogisticRegression')

pickle_classifier(LinearSVC(), X_train, X_test, 'LinearSVC')

pickle_classifier(SGDClassifier(), X_train, X_test, 'SGDClassifier')



