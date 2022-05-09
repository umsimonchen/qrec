# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:32:06 2022

@author: simon
"""

from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
from util.loss import bpr_loss
import numpy as np
import os
from random import sample, choice
from util import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.reset_default_graph()

class ConsisRecplus(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(ConsisRecplus, self).readConfiguration()
        args = config.OptionConf(self.config['ConsisRecplus'])
        self.n_layers = int(args['-n_layer'])
    
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0] # here we do not normalize because we have own score algorithm
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    
    def initModel(self):
        super(ConsisRecplus, self).initModel()
        S = self.buildSparseRelationMatrix()
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape) #social: m*m
        self.S =tf.sparse.to_dense(self.S)
    
    def positive_sample(self, u_i):
        i_i = []
        u = 0
        while u+u_i<self.num_users and u < 100:
            real_u = self.data.id2user[u+u_i]
            positive = list(self.data.trainSet_u[real_u].keys())
            i_i.append(self.data.item[choice(positive)])
            u += 1
        return i_i
    
    def trainModel(self):
        self.positive_i = tf.placeholder(tf.int32, shape=[None])
        self.segment_u = tf.placeholder(tf.int32)
        #self.segment_neighbor = tf.placeholder(tf.int32)
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['query'] = tf.Variable(initializer([2 * self.emb_size, self.emb_size]), name='query')
        for k in range(self.n_layers):
            self.weights['weights%d' % k ] = tf.Variable(initializer([2 * self.emb_size, self.emb_size]), name='weights%d' % k)
        
        def query(n_ego):
            ego, u_embedding = tf.split(tf.reshape(n_ego, [1, 3 * self.emb_size]), [2* self.emb_size, self.emb_size], axis=1)
            q = tf.nn.relu(tf.matmul(ego, self.weights['query']))
            score = tf.math.square(tf.norm(q-u_embedding, axis=1))
            score = tf.math.exp(-score)
            return score[0]
        
        def loop_user(u_ego):
            ego = tf.reshape(u_ego, [1, 2 * self.emb_size])
            neighbors_ego = tf.concat([tf.tile(ego, [self.num_users, 1]), all_user_embeddings[k]], axis=1)
            scores = tf.vectorized_map(fn=lambda em: query(em), elems=neighbors_ego)
            scores = scores / tf.norm(scores, 1)
            return scores
        
        def agg(weights):
            weight = tf.reshape(weights, [1, self.num_users])
            h_neighbor = tf.multiply(weight, tf.transpose(all_user_embeddings[k])) #d*m
            h_neighbor = tf.reduce_sum(h_neighbor, axis=1) #d
            return tf.transpose(h_neighbor)
            
        positive_item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.positive_i)
        users_ego = tf.concat([self.user_embeddings[self.segment_u:self.segment_u+100], positive_item_embeddings], axis=1) #100*2d
        all_user_embeddings = [tf.identity(self.user_embeddings)]
        for k in range(self.n_layers):
            weight_whole = tf.zeros([self.num_users, self.num_users], tf.float32)
            neighbors_weight = tf.vectorized_map(fn=lambda em:loop_user(em), elems=users_ego) #100*m
            weight_whole = tf.concat([weight_whole[:self.segment_u], neighbors_weight, weight_whole[self.segment_u+100:]], axis=0) #m*m
            neighbors_weight = self.S.__mul__(weight_whole)[self.segment_u:self.segment_u+100] #100*m
            h = tf.vectorized_map(fn=lambda em: agg(em), elems=neighbors_weight) #100*d
            print(tf.shape(h))
            #h = tf.multiply(neighbors_weight, all_user_embeddings[k][self.segment_u:self.segment_u+100]) #100*m
            h_ego = tf.concat([all_user_embeddings[k][self.segment_u:self.segment_u+100], h], axis=1) #100*2d
            new_user_embeddings = tf.nn.relu(tf.matmul(h_ego, self.weights['weights%d' % k])) #100*d
            new_user_embeddings = tf.concat([all_user_embeddings[k][:self.segment_u], new_user_embeddings, all_user_embeddings[k][self.segment_u+100:]], axis=0)
            all_user_embeddings.append(new_user_embeddings)
        
        user_embeddings = tf.reduce_sum(all_user_embeddings, axis=0)
        #Prediction
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                i_i = self.positive_sample(u_i)
                _, l, self.U, self.V = self.sess.run([train, loss, user_embeddings, self.item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.segment_u: u_i, self.positive_i: i_i})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
            self.ranking_performance(epoch)
        self.U, self.V = self.bestU, self.bestV
        
    def saveModel(self):
        self.bestU = self.U
        self.bestV = self.V
        
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        #print(type(self.V),type(self.U))
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            #print(self.V.dot(self.U[u]))
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
    
    
 
        
    