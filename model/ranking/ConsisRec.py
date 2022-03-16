# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:21:29 2022

@author: simon
"""
#import tensorflow as tf
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
from util.loss import bpr_loss
import numpy as np
import os
from util import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.reset_default_graph()

class ConsisRec(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(ConsisRec, self).readConfiguration()
        args = config.OptionConf(self.config['ConsisRec'])
        self.n_layers = int(args['-n_layer'])
    
    def trainItemData(self, i):
        itemSegment = i
        buyers = []
        for buyer in self.data.trainSet_i[self.data.id2item[itemSegment]].keys():
            buyers.append(self.data.user[buyer])
        buyers_length = len(buyers)
        return itemSegment, buyers, buyers_length
    
    def trainUserData(self, i):
        userSegment = i
        neighbors = []
        bought_items = []
        for neighbor in self.social.followees[self.data.id2user[userSegment]].keys():
            neighbors.append(self.data.user[neighbor])
        for neighbor in self.social.followers[self.data.id2user[userSegment]].keys():
            neighbors.append(self.data.user[neighbor])
        neighbors_length = len(list(set(neighbors)))    
        
        for item in self.data.trainSet_u[self.data.id2user[userSegment]].keys():
            bought_items.append(self.data.item[item])
        bought_length = len(bought_items)
            
        return userSegment, neighbors, neighbors_length, bought_items, bought_length
    
    def initModel(self):
        super(ConsisRec, self).initModel()
                  
    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        user_item_map = np.full((self.num_users, self.num_items), -1, dtype=np.int32)
        user_item_index = 0
        for user in self.data.trainSet_u.keys():
            for item in self.data.trainSet_u[user].keys():
                user_item_map[self.data.user[user]][self.data.item[item]] = user_item_index
                user_item_index += 1
        user_item_map = tf.convert_to_tensor(user_item_map)
        user_item_embeddings = tf.Variable(tf.truncated_normal(shape=[user_item_index+1, self.emb_size], stddev=0.005), name='U_U')
        
        user_user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.num_users, self.emb_size], stddev=0.005), name='U_U')
        
        self.weights['query_weights'] = tf.Variable(
            initializer([2 * self.emb_size, self.emb_size]), name='query_weights')
        self.weights['attention_weights'] = tf.Variable(
            initializer([2 * self.emb_size, 1]), name='attention_weights')
        
        all_user_embeddings = []
        all_item_embeddings = []
        for k in range(self.n_layers+1):
            all_user_embeddings.append(tf.identity(self.user_embeddings)) # m*d
            all_item_embeddings.append(tf.identity(self.item_embeddings)) # n*d
        
        #item diffusion tensor initialization
        self.item_index = tf.placeholder(tf.int32, shape=[], name="item_index")
        self.buyers_length = tf.placeholder(tf.int32, shape=[], name="buyers_length")
        self.buyers = tf.placeholder(tf.int32, shape=[None,], name="buyers")
                
        #user diffusion tensor initialization
        self.user_index = tf.placeholder(tf.int32, shape=[], name="user_index")
        self.neighbors_length = tf.placeholder(tf.int32, shape=[], name="neighbors_length")
        self.neighbors = tf.placeholder(tf.int32, shape=[None,], name="neighbors")
        self.bought_length = tf.placeholder(tf.int32, shape=[], name="bought_length")
        self.bought_items = tf.placeholder(tf.int32, shape=[None,], name="bought_items")
        
        for k in range(self.n_layers):
            self.weights['neighbor_sample%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, self.emb_size]), name='neighbor_sample%d' % k)
           
        for k in range(self.n_layers):
            #item
            hv_item = tf.reshape(tf.nn.embedding_lookup(all_item_embeddings[k], self.item_index), [1, self.emb_size])
            hi_user = tf.nn.embedding_lookup(all_user_embeddings[k], self.buyers) #j*d
            qi_user = tf.concat([tf.tile(hv_item, [self.buyers_length,1]), hi_user], 1) #j*2d
            qi_user = tf.matmul(qi_user, self.weights['query_weights']) #j*d
            si_user = tf.norm(qi_user-hi_user, axis=1) #j*1 
            si_user = tf.transpose(tf.math.exp(-si_user)) #1*j
            si_user = si_user / tf.norm(si_user,1) #norm 1
            new_hi_user = tf.multiply(si_user, tf.transpose(hi_user)) #d*j
            
            ei_index = tf.stack([self.buyers, tf.tile([self.item_index], [self.buyers_length])], axis=1) #j*2
            ei_emb_index = tf.gather_nd(user_item_map, ei_index)
            ei = tf.nn.embedding_lookup(user_item_embeddings, ei_emb_index) #j*d
            ei_ego = tf.concat([hi_user, ei], 1) #j*2d
            ai = tf.matmul(ei_ego, self.weights['attention_weights']) #j*1
            ai = tf.nn.softmax(tf.transpose(ai)) #1*j
            
            new_hi_user = tf.reduce_sum(tf.transpose(tf.multiply(ai, new_hi_user)), 0) #d*j -> j*d -> d
            hi_ego = tf.concat([hv_item, [new_hi_user]], 1)
            new_hv_item = tf.nn.relu(tf.matmul(hi_ego, self.weights['neighbor_sample%d' % k]))
            all_item_embeddings[k+1] = tf.concat([all_item_embeddings[k+1][:self.item_index], new_hv_item, all_item_embeddings[k+1][self.item_index+1:]], 0)
            
        self.user_embeddings = all_user_embeddings[self.n_layers]
        self.item_embeddings = all_item_embeddings[self.n_layers]
        
        #Prediction
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u_idx)
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
                for i in range(100):
                    itemSegment, buyers, buyers_length = self.trainItemData(i_idx[i])
                    userSegment, neighbors, neighbors_length, bought_items, bought_length = self.trainUserData(user_idx[i])
                    self.U, self.V = self.sess.run([self.user_embeddings, self.item_embeddings], 
                                                   feed_dict={self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers, \
                                                              self.user_index: userSegment, self.neighbors_length: neighbors_length, self.neighbors: neighbors, \
                                                              self.bought_length: bought_length, self.bought_items: bought_items})
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, 
                                                self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers,\
                                                    self.user_index: userSegment, self.neighbors_length: neighbors_length, self.neighbors: neighbors, \
                                                    self.bought_length: bought_length, self.bought_items: bought_items})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
            self.ranking_performance(epoch)
        self.U,self.V = self.bestU,self.bestV
        
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
    
    
 
        
    