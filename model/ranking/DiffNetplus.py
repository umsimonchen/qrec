# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:57:27 2022

@author: simon
"""

from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
import os
from util import config
from random import choice
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.reset_default_graph()
#For general comparison. We do not include the user/item features extracted from text/images

class DiffNetplus(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
    
    def trainItemData(self, i):
        itemSegment = i
        buyers = []
        buyers_length = 0
        for buyer in self.data.trainSet_i[self.data.id2item[itemSegment]].keys():
            buyers.append(self.data.user[buyer])
            buyers_length += 1
        return itemSegment, buyers, buyers_length
    
    def trainUserData(self, i):
        userSegment = i
        followees = []
        followees_length = 0
        bought_items = []
        bought_length = 0
        for followee in self.social.followees[self.data.id2user[userSegment]].keys():
            followees.append(self.data.user[followee])
            followees_length += 1
        for item in self.data.trainSet_u[self.data.id2user[userSegment]].keys():
            bought_items.append(self.data.item[item])
            bought_length += 1
            
        return userSegment, followees, followees_length, bought_items, bought_length
    
    def readConfiguration(self):
        super(DiffNetplus, self).readConfiguration()
        args = config.OptionConf(self.config['DiffNetplus'])
        self.n_layers = int(args['-n_layer']) #the number of layers of the recommendation module (discriminator)
    
    def next_batch_pairwise_one_out(self):
        batch_id = 0
        batch_end = len(self.data.trainingData)
        while batch_id < batch_end:
            users = [self.data.trainingData[idx][0] for idx in range(batch_end) if idx != batch_id] # real user id
            items = [self.data.trainingData[idx][1] for idx in range(batch_end) if idx != batch_id] # real item id
                     
            u_idx, i_idx, j_idx = [], [], []
            self.data.testSet_u = {self.data.trainingData[batch_id][0]: {self.data.trainingData[batch_id][0]:1.0}}
            item_list = list(self.data.item.keys())
            for i, user in enumerate(users):
                i_idx.append(self.data.item[items[i]]) # training item id, positive since training data recorded
                u_idx.append(self.data.user[user]) # training user id
                neg_item = choice(item_list) # random choose one as negative sample with real user/item id
                # make sure it negative sample or replace it
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])
                
            yield u_idx, i_idx, j_idx # return a batch of training data
            
            batch_id += 1
            
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0/len(self.social.followees[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    
    def initModel(self):
        super(DiffNetplus, self).initModel()
        S = self.buildSparseRelationMatrix()
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape) #social: m*m
        self.A = self.create_sparse_adj_tensor() #item: m*n, normalized

    def trainModel(self):
        self.weights = {}
        all_user_embeddings = []
        all_item_embeddings = []
        for k in range(self.n_layers+1):
            all_user_embeddings.append(tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.01), name='U%d' % k)) # m*d
            all_item_embeddings.append(tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.01), name='V%d' % k)) # n*d
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_fusion_weights'] = tf.Variable(
            initializer([self.emb_size, self.emb_size]), name='user_fusion_weights')
        self.weights['item_fusion_weights'] = tf.Variable(
            initializer([self.emb_size, self.emb_size]), name='item_fusion_weights')
        all_user_embeddings[0] = tf.matmul(all_user_embeddings[0], self.weights['user_fusion_weights']) #m*d
        all_item_embeddings[0] = tf.matmul(all_item_embeddings[0], self.weights['item_fusion_weights']) #m*d
        
        #item diffusion tensor initialization
        self.item_index = tf.placeholder(tf.int32, shape=[], name="item_index")
        self.buyers_length = tf.placeholder(tf.int32, shape=[], name="buyers_length")
        self.buyers = tf.placeholder(tf.int32, shape=[None,], name="buyers")
                
        #user diffusion tensor initialization
        self.user_index = tf.placeholder(tf.int32, shape=[], name="user_index")
        self.followees_length = tf.placeholder(tf.int32, shape=[], name="followees_length")
        self.followees = tf.placeholder(tf.int32, shape=[None,], name="followees")
        self.bought_length = tf.placeholder(tf.int32, shape=[], name="bought_length")
        self.bought_items = tf.placeholder(tf.int32, shape=[None,], name="bought_items")
                
        #Layers
        for k in range(self.n_layers):
            self.weights['mlp_one_in_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 2 * self.emb_size]), name='mlp_one_input_weights%d' % k)
            self.weights['mlp_one_out_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 1]), name='mlp_one_out_weights%d' % k)
            self.weights['mlp_two_in_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 2 * self.emb_size]), name='mlp_two_input_weights%d' % k)
            self.weights['mlp_two_out_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 1]), name='mlp_two_out_weights%d' % k)
            self.weights['mlp_three_in_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 2 * self.emb_size]), name='mlp_three_input_weights%d' % k)
            self.weights['mlp_three_out_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 1]), name='mlp_three_out_weights%d' % k)
            self.weights['mlp_four_in_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 2 * self.emb_size]), name='mlp_four_input_weights%d' % k)
            self.weights['mlp_four_out_weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, 1]), name='mlp_four_out_weights%d' % k)
            
            #Item diffusion
            item_embedding = tf.reshape(tf.nn.embedding_lookup(all_item_embeddings[k], self.item_index), [1, self.emb_size])
            buyers_embedding = tf.nn.embedding_lookup(all_user_embeddings[k], self.buyers)
            
            vu_ego_embedding = tf.concat([tf.tile(item_embedding, [self.buyers_length,1]), buyers_embedding], 1)
            vu_ego_embedding = tf.matmul(vu_ego_embedding,self.weights['mlp_one_in_weights%d' % k])
            vu_ego_embedding = tf.matmul(vu_ego_embedding,self.weights['mlp_one_out_weights%d' % k])
            vu_ego_embedding = tf.nn.relu(vu_ego_embedding)
            eta = tf.nn.softmax(tf.transpose(vu_ego_embedding))
            
            new_item_embedding = tf.matmul(eta,buyers_embedding)
            new_item_embedding = tf.math.l2_normalize(new_item_embedding,axis=1)
            all_item_embeddings[k+1] = tf.concat([all_item_embeddings[k+1][:self.item_index], new_item_embedding, all_item_embeddings[k+1][self.item_index+1:]], 0)
            
            #User diffusion
            user_embedding = tf.reshape(tf.nn.embedding_lookup(all_user_embeddings[k], self.user_index), [1, self.emb_size])
            followees_embedding = tf.nn.embedding_lookup(all_user_embeddings[k], self.followees)
            bought_items_embedding = tf.nn.embedding_lookup(all_item_embeddings[k], self.bought_items)
            
            #User neighbor
            uu_ego_embedding = tf.concat([tf.tile(user_embedding, [self.followees_length,1]), followees_embedding], 1)
            uu_ego_embedding = tf.matmul(uu_ego_embedding,self.weights['mlp_two_in_weights%d' % k])
            uu_ego_embedding = tf.matmul(uu_ego_embedding,self.weights['mlp_two_out_weights%d' % k])
            uu_ego_embedding = tf.nn.relu(uu_ego_embedding)
            alpha = tf.nn.softmax(tf.transpose(uu_ego_embedding))
            
            user_neighbors_embedding = tf.matmul(alpha,followees_embedding)
            
            #Item neighbor
            uv_ego_embedding = tf.concat([tf.tile(user_embedding, [self.bought_length,1]), bought_items_embedding], 1)
            uv_ego_embedding = tf.matmul(uv_ego_embedding,self.weights['mlp_three_in_weights%d' % k])
            uv_ego_embedding = tf.matmul(uv_ego_embedding,self.weights['mlp_three_out_weights%d' % k])
            uv_ego_embedding = tf.nn.relu(uv_ego_embedding)
            beta = tf.nn.softmax(tf.transpose(uv_ego_embedding))
            
            item_neighbors_embedding = tf.matmul(beta,bought_items_embedding)
            
            #Aggregation
            up_ego_embedding = tf.concat([user_embedding, user_neighbors_embedding], 1)
            uq_ego_embedding = tf.concat([user_embedding, item_neighbors_embedding], 1)
            upq_ego_embedding = tf.concat([up_ego_embedding, uq_ego_embedding], 0)
            upq_ego_embedding = tf.matmul(upq_ego_embedding,self.weights['mlp_four_in_weights%d' % k])
            upq_ego_embedding = tf.matmul(upq_ego_embedding,self.weights['mlp_four_out_weights%d' % k])
            upq_ego_embedding = tf.nn.relu(upq_ego_embedding)
            gamma = tf.nn.softmax(tf.reshape(upq_ego_embedding, [1, 2]))
            
            new_user_embedding = tf.matmul(gamma,tf.concat([user_neighbors_embedding, item_neighbors_embedding], 0))
            new_user_embedding = tf.math.l2_normalize(new_user_embedding,axis=1)
            all_user_embeddings[k+1] = tf.concat([all_user_embeddings[k+1][:self.user_index], new_user_embedding, all_user_embeddings[k+1][self.user_index+1:]], 0) 
        
        #Organization
        self.user_embeddings = tf.identity(all_user_embeddings[0])
        self.item_embeddings = tf.identity(all_item_embeddings[0])
        for k in range(self.n_layers):
            self.user_embeddings = tf.concat([self.user_embeddings, all_user_embeddings[k+1]], 1)
            self.item_embeddings = tf.concat([self.item_embeddings, all_item_embeddings[k+1]], 1)
            
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
                for i in range(1): # i is hyperparameter, 1 and 10 are optimal
                    itemSegment, buyers, buyers_length = self.trainItemData(i_idx[i])
                    userSegment, followees, followees_length, bought_items, bought_length = self.trainUserData(user_idx[i])
                    self.U, self.V = self.sess.run([self.user_embeddings, self.item_embeddings], 
                                                   feed_dict={self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers, \
                                                              self.user_index: userSegment, self.followees_length: followees_length, self.followees: followees, \
                                                              self.bought_length: bought_length, self.bought_items: bought_items})
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, 
                                                self.item_index: itemSegment, self.buyers_length: buyers_length, self.buyers: buyers,\
                                                    self.user_index: userSegment, self.followees_length: followees_length, self.followees: followees, \
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