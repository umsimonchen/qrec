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
from random import sample 
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
        buyers = sample(buyers, max(1, int(0.5*len(buyers))))
        buyers_length = len(buyers)
        
        item_neighbors = []
        if self.data.id2item[itemSegment] in self.item_item.keys():
            for item in self.item_item[self.data.id2item[itemSegment]]:
                if item in self.data.item.keys(): # sometimes the linked item not in such training data
                    item_neighbors.append(self.data.item[item])
        item_neighbors_len = len(item_neighbors)
        
        return itemSegment, item_neighbors, item_neighbors_len, buyers, buyers_length
    
    def trainUserData(self, i):
        userSegment = i
        neighbors = []
        bought_items = []
        for neighbor in self.social.followees[self.data.id2user[userSegment]].keys():
            neighbors.append(self.data.user[neighbor])
        for neighbor in self.social.followers[self.data.id2user[userSegment]].keys():
            neighbors.append(self.data.user[neighbor])
        neighbors = list(set(neighbors))
        neighbors = sample(neighbors, max(1, int(0.5*len(neighbors))))
        neighbors_length = len(neighbors)    
        
        for item in self.data.trainSet_u[self.data.id2user[userSegment]].keys():
            bought_items.append(self.data.item[item])           
        bought_items = sample(bought_items, max(1, int(0.5*len(bought_items))))
        bought_length = len(bought_items)
            
        return userSegment, neighbors, neighbors_length, bought_items, bought_length
            
    def initModel(self):
        super(ConsisRec, self).initModel()
        self.item_item = {} #real item id
        with open('./././dataset/lastfm/itemlinks.txt', 'r') as f:
            for line in f.readlines():
                pair = line.split()
                if pair[0] in self.data.item.keys() and pair[1] in self.data.item.keys():
                    if pair[0] not in self.item_item.keys():
                        self.item_item[pair[0]] = [pair[1]]
                    else:
                        self.item_item[pair[0]].append(pair[1])
        
    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        
        #embedding relation preprocess
        user_item_map = np.full((self.num_users, self.num_items), -1, dtype=np.int32)
        user_item_index = 0
        for user in self.data.trainSet_u.keys():
            for item in self.data.trainSet_u[user].keys():
                user_item_map[self.data.user[user]][self.data.item[item]] = user_item_index
                user_item_index += 1
        user_item_map = tf.convert_to_tensor(user_item_map)
        user_item_embeddings = tf.Variable(tf.truncated_normal(shape=[user_item_index+1, self.emb_size], stddev=0.005), name='U_V')
        
        user_user_map = np.full((self.num_users, self.num_users), -1, dtype=np.int32)
        user_user_index = 0
        for user in self.social.followees.keys():
            for followee in self.social.followees[user].keys():
                user_user_map[self.data.user[user]][self.data.user[followee]] = user_user_index
                user_user_index += 1
        user_user_map = user_user_map + user_user_map.T + np.full((self.num_users, self.num_users), 1, dtype=np.int32)
        user_user_map = tf.convert_to_tensor(user_user_map)
        user_user_embeddings = tf.Variable(tf.truncated_normal(shape=[user_user_index+1, self.emb_size], stddev=0.005), name='U_U')
        
        self.weights['query_weights'] = tf.Variable(
            initializer([2 * self.emb_size, self.emb_size]), name='query_weights')
        self.weights['attention_weights'] = tf.Variable(
            initializer([2 * self.emb_size, 1]), name='attention_weights')
        
        item_item_map = np.full((self.num_items, self.num_items), -1, dtype=np.int32)
        item_item_index = 0
        for item1 in self.item_item.keys() :
            for item2 in self.item_item[item1]:
                if int(item1) < int(item2):
                    item_item_map[self.data.item[item1]][self.data.item[item2]] = item_item_index
                    item_item_index += 1
        item_item_map = item_item_map + item_item_map.T + np.full((self.num_items, self.num_items), 1, dtype=np.int32)
        item_item_map = tf.convert_to_tensor(item_item_map)
        item_item_embeddings = tf.Variable(tf.truncated_normal(shape=[item_item_index+1, self.emb_size], stddev=0.005), name='V_V')
             
        all_user_embeddings = []
        all_item_embeddings = []
        for k in range(self.n_layers+1):
            all_user_embeddings.append(tf.identity(self.user_embeddings)) # m*d
            all_item_embeddings.append(tf.identity(self.item_embeddings)) # n*d
        
        #item diffusion tensor initialization
        self.item_index = tf.placeholder(tf.int32, shape=[], name="item_index")
        self.buyers_length = tf.placeholder(tf.int32, shape=[], name="buyers_length")
        self.buyers = tf.placeholder(tf.int32, shape=[None,], name="buyers")
        self.item_neighbors_len = tf.placeholder(tf.int32, shape=[], name="item_neighbors_len")    
        self.item_neighbors = tf.placeholder(tf.int32, shape=[None], name="item_neighbors")
        
        #user diffusion tensor initialization
        self.user_index = tf.placeholder(tf.int32, shape=[], name="user_index")
        self.neighbors_length = tf.placeholder(tf.int32, shape=[], name="neighbors_length")
        self.neighbors = tf.placeholder(tf.int32, shape=[None,], name="neighbors")
        self.bought_length = tf.placeholder(tf.int32, shape=[], name="bought_length")
        self.bought_items = tf.placeholder(tf.int32, shape=[None,], name="bought_items")
        
        for k in range(self.n_layers):
            self.weights['neighbor_sample%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, self.emb_size]), name='neighbor_sample%d' % k)
        
        def only_user_item():
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
            return new_hv_item
        
        def item_item():
            hv_item = tf.reshape(tf.nn.embedding_lookup(all_item_embeddings[k], self.item_index), [1, self.emb_size])
            hi_user = tf.nn.embedding_lookup(all_user_embeddings[k], self.buyers) #j*d
            hi_item_neighbor = tf.nn.embedding_lookup(all_item_embeddings[k], self.item_neighbors)
            hi_item_ego = tf.concat([hi_item_neighbor, hi_user], 0)
            positive_user = tf.nn.embedding_lookup(all_user_embeddings[k], [self.buyers[0]])
            
            qi_user = tf.concat([hi_user, tf.tile(hv_item, [self.buyers_length, 1])], 1) #j*2d
            qi_item_neighbor = tf.concat([tf.tile(positive_user, [self.item_neighbors_len, 1]), hi_item_neighbor], 1)
            qi_item_ego = tf.concat([qi_item_neighbor, qi_user], 0)
            qi_item_ego = tf.matmul(qi_item_ego, self.weights['query_weights']) #j*d
            si_item_ego = tf.math.square(tf.norm(qi_item_ego-hi_item_ego, axis=1)) #j*1 
            si_item_ego = tf.transpose(tf.math.exp(-si_item_ego)) #1*j
            si_item_ego = si_item_ego / (tf.norm(si_item_ego,1)+1e-8) #norm 1
            new_hi_item_ego = tf.multiply(si_item_ego, tf.transpose(hi_item_ego)) #d*j
            
            ei_user_index = tf.stack([self.buyers, tf.tile([self.item_index], [self.buyers_length])], axis=1) #j*2
            ei_user_emb_index = tf.gather_nd(user_item_map, ei_user_index)
            ei_user = tf.nn.embedding_lookup(user_item_embeddings, ei_user_emb_index) #j*d
            ei_neighbor_index = tf.stack([tf.tile([self.item_index], [self.item_neighbors_len]), self.item_neighbors], axis=1)
            ei_neighbor_emb_index = tf.gather_nd(item_item_map, ei_neighbor_index)
            ei_neighbor = tf.nn.embedding_lookup(item_item_embeddings, ei_neighbor_emb_index)
            ei_item_ego = tf.concat([ei_user, ei_neighbor], 0)
            ei_item_ego = tf.concat([hi_item_ego, ei_item_ego], 1) #j*2d
            ai_item = tf.matmul(ei_item_ego, self.weights['attention_weights']) #j*1
            ai_item = tf.nn.softmax(tf.transpose(ai_item)) #1*j
            
            new_hi_item_ego = tf.reduce_sum(tf.transpose(tf.multiply(ai_item, new_hi_item_ego)), 0) #d*j -> j*d -> d
            new_hi_item_ego = tf.concat([hv_item, [new_hi_item_ego]], 1)
            new_hv_item = tf.nn.relu(tf.matmul(new_hi_item_ego, self.weights['neighbor_sample%d' % k]))
            return new_hv_item
        
        for k in range(self.n_layers):
            #item
            new_hv_item = tf.cond(tf.equal(self.item_neighbors_len, 0), lambda: only_user_item(), lambda: only_user_item())   
            all_item_embeddings[k+1] = tf.concat([all_item_embeddings[k+1][:self.item_index], new_hv_item, all_item_embeddings[k+1][self.item_index+1:]], 0)
            
            #user
            hv_user = tf.reshape(tf.nn.embedding_lookup(all_user_embeddings[k], self.user_index), [1, self.emb_size])
            hi_item = tf.nn.embedding_lookup(all_item_embeddings[k], self.bought_items)

# Only user_item           
# =============================================================================
#             qi_item = tf.concat([tf.tile(hv_user, [self.bought_length, 1]), hi_item], 1)
#             qi_item = tf.matmul(qi_item, self.weights['query_weights'])
#             si_item = tf.math.square(tf.norm(qi_item-hi_item, axis=1))
#             si_item = tf.transpose(tf.math.exp(-si_item))
#             si_item = si_item / (tf.norm(si_item,1)+1e-8)
#             new_hi_item = tf.multiply(si_item, tf.transpose(hi_item))
#             
#             ei_index = tf.stack([tf.tile([self.user_index], [self.bought_length]), self.bought_items], axis=1)
#             ei_emb_index = tf.gather_nd(user_item_map, ei_index)
#             ei = tf.nn.embedding_lookup(user_item_embeddings, ei_emb_index)
#             ei_ego = tf.concat([hi_item, ei], 1)
#             ai = tf.matmul(ei_ego, self.weights['attention_weights'])
#             ai = tf.nn.softmax(tf.transpose(ai))
#             
#             new_hi_item = tf.reduce_sum(tf.transpose(tf.multiply(ai, new_hi_item)), 0)
#             new_hi_item = tf.concat([hv_user, [new_hi_item]], 1)
#             new_hv_user = tf.nn.relu(tf.matmul(new_hi_item, self.weights['neighbor_sample%d' % k]))
#             all_user_embeddings[k+1] = tf.concat([all_user_embeddings[k+1][:self.user_index], new_hv_user, all_user_embeddings[k+1][self.user_index+1:]], 0)
# =============================================================================
            
            hi_user_neighbor = tf.nn.embedding_lookup(all_user_embeddings[k], self.neighbors)
            hi_user_ego = tf.concat([hi_item, hi_user_neighbor], 0)
            positive_item = tf.nn.embedding_lookup(all_item_embeddings[k], [self.bought_items[0]]) #choose one positive item for neighbor
            
            qi_item = tf.concat([tf.tile(hv_user, [self.bought_length, 1]), hi_item], 1)
            qi_user_neighbor = tf.concat([hi_user_neighbor, tf.tile(positive_item, [self.neighbors_length, 1])], 1)
            qi_user_ego = tf.concat([qi_item, qi_user_neighbor], 0) #(j+q)*2d
            qi_user_ego = tf.matmul(qi_user_ego, self.weights['query_weights']) #(j+q)*d
            si_user_ego = tf.math.square((tf.norm(qi_user_ego-hi_user_ego, axis=1))) #(j+q)*1
            si_user_ego = tf.transpose(tf.math.exp(-si_user_ego)) #1*(j+q)
            si_user_ego = si_user_ego / (tf.norm(si_user_ego,1)+1e-8)
            new_hi_user_ego = tf.multiply(si_user_ego, tf.transpose(hi_user_ego)) #d*(j+q)
            
            ei_item_index = tf.stack([tf.tile([self.user_index], [self.bought_length]), self.bought_items], axis=1)
            ei_item_emb_index = tf.gather_nd(user_item_map,ei_item_index)
            ei_item = tf.nn.embedding_lookup(user_item_embeddings, ei_item_emb_index)
            ei_neighbor_index = tf.stack([tf.tile([self.user_index], [self.neighbors_length]), self.neighbors], axis=1)
            ei_neighbor_emb_index = tf.gather_nd(user_user_map, ei_neighbor_index)
            ei_neighbor = tf.nn.embedding_lookup(user_user_embeddings, ei_neighbor_emb_index)
            ei_user_ego = tf.concat([ei_item, ei_neighbor], 0) #(j+q)*d
            ei_user_ego = tf.concat([hi_user_ego, ei_user_ego], 1) #(j+q)*2d
            ai_user = tf.matmul(ei_user_ego, self.weights['attention_weights']) #(j+d)*1
            ai_user = tf.nn.softmax(tf.transpose(ai_user)) #1*(j+q)
            
            new_hi_user_ego = tf.reduce_sum(tf.transpose(tf.multiply(ai_user, new_hi_user_ego)), 0)
            new_hi_user_ego = tf.concat([hv_user, [new_hi_user_ego]], 1)
            new_hv_user = tf.nn.relu(tf.matmul(new_hi_user_ego, self.weights['neighbor_sample%d' % k]))
            
            all_user_embeddings[k+1] = tf.concat([all_user_embeddings[k+1][:self.user_index], new_hv_user, all_user_embeddings[k+1][self.user_index+1:]], 0)
              
        self.user_embeddings = tf.identity(all_user_embeddings[self.n_layers])
        self.item_embeddings = tf.identity(all_item_embeddings[self.n_layers])
        
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
                for i in range(1):
                    itemSegment, item_neighbors, item_neighbors_len, buyers, buyers_length = self.trainItemData(i_idx[i])
                    userSegment, neighbors, neighbors_length, bought_items, bought_length = self.trainUserData(user_idx[i])
                    self.U, self.V = self.sess.run([self.user_embeddings, self.item_embeddings], 
                                                   feed_dict={self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers, \
                                                              self.item_neighbors: item_neighbors, self.item_neighbors_len: item_neighbors_len, \
                                                              self.user_index: userSegment, self.neighbors_length: neighbors_length, self.neighbors: neighbors, \
                                                              self.bought_length: bought_length, self.bought_items: bought_items})
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, 
                                                self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers,\
                                                    self.item_neighbors: item_neighbors, self.item_neighbors_len: item_neighbors_len, \
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
    
    
 
        
    