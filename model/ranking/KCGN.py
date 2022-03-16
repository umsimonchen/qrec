from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
from random import choice
import tensorflow as tf
from scipy.sparse import coo_matrix
from math import sqrt
import numpy as np
import os
from util import config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ESRF(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
        self.n_layers = 2 #the number of layers of the alternative neigbhor generation module (generator)
    