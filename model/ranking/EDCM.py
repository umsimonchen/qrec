# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:48:11 2022

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
import pandas as pd
from math import sqrt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.reset_default_graph()

class EDCM(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(EDCM, self).readConfiguration()
        args = config.OptionConf(self.config['EDCM'])
        self.n_layer = int(args['-n_layer'])
        self.n_channel = int(args['-n_channel'])
        self.neighbor_percent = float(args['-neighbor_percent'])
        self.temperature = float(args['-temperature'])
        self.K = int(args['-K'])
    
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0/len(self.social.followers[pair[0]])] # here we normalize
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0/len(self.data.trainSet_u[pair[0]])]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix
    
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    # def buildSparseHeteroMatrix(self):
    #     row, col, entries = [], [], []
    #     for pair in self.social.relation:
    #         row += [self.data.user[pair[0]]]
    #         col += [self.data.user[pair[1]]]
    #         entries += [1.0] 
    #     for pair in self.data.trainingData:
    #         #down-ward
    #         row += [self.num_users+self.data.item[pair[1]]]
    #         col += [self.data.user[pair[0]]]
    #         entries += [1.0]
    #         #right-ward
    #         row += [self.data.user[pair[0]]]
    #         col += [self.num_users+self.data.item[pair[1]]]
    #         entries += [1.0]
    #     heteroMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users+self.num_items,self.num_users+self.num_items),dtype=np.float32)
    #     return heteroMatrix    
    
    def initModel(self):
        super(EDCM, self).initModel()
        self.S = self.buildSparseRelationMatrix() 
        self.S_one = (self.S>0).astype(dtype=np.float32) 
        self.S_zero = (self.S<1e-10).astype(dtype=np.float32)
        
        self.Y = self.buildSparseRatingMatrix()
        
        self.Y_label = ((self.Y>0).astype(dtype=np.float32)).dot((self.Y).T)
        self.Y_homo = ((self.Y_label>0.5).astype(dtype=np.float32)).multiply(self.S_one)
        self.Y_hetero = ((self.Y_label<=0.5).astype(dtype=np.float32)).multiply(self.S_one)
        
        self.S_one = self.adj_to_sparse_tensor(self.S_one)
        self.S_zero = self.adj_to_sparse_tensor(self.S_zero)
        self.Y = self.adj_to_sparse_tensor(self.Y)
        self.Y_homo = self.adj_to_sparse_tensor(self.Y_homo)
        self.Y_hetero = self.adj_to_sparse_tensor(self.Y_hetero)
        
    def positive_sample(self, u_i):
        i_i = []
        u = 0
        while u+u_i<self.num_users and u < 100:
            real_u = self.data.id2user[u+u_i]
            positive = list(self.data.trainSet_u[real_u].keys())
            i_i.append(self.data.item[choice(positive)])
            u += 1
        return i_i
    
    def positvie_sample_all(self):
        i_i = []
        for i in list(self.data.trainSet_u.keys()):
            positive = list(self.data.trainSet_u[i].keys())
            i_i.append(self.data.item[choice(positive)])
        return i_i
    
    def sampling(self, a):
        eps = 1e-10
        a = tf.nn.softmax(a) #k*m
        u = tf.random_uniform(tf.shape(a))
        gumbel_noise = -tf.log(-tf.log(u+eps)+eps)
        y = tf.log(a+eps) + gumbel_noise
        return tf.nn.softmax(y / self.temperature)
        
    def trainModel(self):
        self.positive_i = tf.placeholder(tf.int32, shape=[None])
        self.segment_u = tf.placeholder(tf.int32)
        self.isOn = tf.placeholder(tf.int32)
        self.weights = {}
        self.d_weights = {}
        self.r_weights = {}
        self.channel_G = {}
        initializer = tf.contrib.layers.xavier_initializer()
        all_user_embeddings = [tf.identity(self.user_embeddings)]
        all_item_embeddings = [tf.identity(self.item_embeddings)]

        for i in range(self.n_layer):
            for j in range(self.n_channel):
                self.weights['w1%d_%d'%(i,j)] = tf.Variable(initializer([2 * self.emb_size, 1]), name='w1%d_%d'%(i,j))
                self.weights['w2%d_%d'%(i,j)] = tf.Variable(initializer([self.num_users, 1]), name='w2%d_%d'%(i,j))
                self.weights['feat%d_%d'%(i,j)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='feat%d_%d'%(i,j))
            self.weights['agg%d'%i] = tf.Variable(initializer([self.n_channel * self.emb_size, self.emb_size]), name='agg%d'%i)
        self.r_weights['recommend1'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='recommend1')
        #self.r_weights['recommend2'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='recommend2')
        self.d_weights['discriminator1'] = tf.Variable(initializer([2 * self.emb_size, 2 * self.emb_size]), name='discriminator1')
        self.d_weights['discriminator2'] = tf.Variable(initializer([2 * self.emb_size, self.n_channel]), name='discriminator2')
        
        def loop_user(users):
            u_embedding = tf.reshape(users, [1, self.emb_size]) 
            e_embedding = tf.concat([tf.tile(u_embedding, [self.num_users, 1]), self.user_embeddings], axis=1) #m*2d
            possibility = tf.matmul(e_embedding, self.weights['w1%d_%d'%(i,j)])
            possibility = tf.nn.relu(possibility) #m*1 ---------shoule be leaky relu-----------------
            possibility = tf.multiply(possibility, self.weights['w2%d_%d'%(i,j)]) #m*1
            return tf.reduce_sum(possibility, axis=1) #m
        
        user_set = self.user_embeddings[self.segment_u:self.segment_u+100]
        eps = 1e-10
        loss_edge = 0
        loss_homo = 0
        loss_hetero = 0
        loss_channel = 0
        for i in range(self.n_layer):  
            for j in range(self.n_channel):
                raw_possibility = tf.vectorized_map(fn=lambda em:loop_user(em), elems=user_set) #m+*m
                channel_possibility = tf.nn.softmax(tf.nn.sigmoid(raw_possibility))
                new_user_embeddings_c = tf.matmul(all_user_embeddings[i], self.weights['feat%d_%d'%(i,j)]) #m*d
                new_user_embeddings_c = tf.matmul(channel_possibility, all_user_embeddings[i])
                semantic_embeddings = tf.concat([all_user_embeddings[i][self.segment_u:self.segment_u+100], new_user_embeddings_c], axis=1)
                predicted = tf.nn.sigmoid(tf.matmul(semantic_embeddings, self.d_weights['discriminator1']))
                predicted = tf.nn.softmax(tf.matmul(predicted, self.d_weights['discriminator2']))
                loss_channel += -tf.reduce_sum(predicted)
                
                if j == 0:
                    new_user_embeddings = new_user_embeddings_c
                    all_raw_possibility = raw_possibility
                    all_homo_possibility = raw_possibility
                    all_hetero_possibility = 0
                else:
                    new_user_embeddings = tf.concat([new_user_embeddings, new_user_embeddings_c], axis=1)
                    all_raw_possibility += raw_possibility 
                    if j < self.n_channel/2:
                        all_homo_possibility += raw_possibility
                    else:
                        all_hetero_possibility += raw_possibility
            
            new_user_embeddings = tf.nn.sigmoid(tf.matmul(new_user_embeddings, self.weights['agg%d'%i]))
            new_user_embeddings = tf.nn.l2_normalize(new_user_embeddings, axis=1)
            
            all_user_embeddings.append(tf.concat([all_user_embeddings[i][:self.segment_u], new_user_embeddings, all_user_embeddings[i][self.segment_u+100:]], axis=0))
            all_item_embeddings.append(self.item_embeddings)
            all_channel_possibility = tf.nn.sigmoid(all_raw_possibility)
            all_homo_possibility = tf.nn.sigmoid(all_homo_possibility)
            all_hetero_possibility = tf.nn.sigmoid(all_hetero_possibility)
            
            s_one_sub = tf.sparse.to_dense(tf.sparse_slice(self.S_one, [self.segment_u,0], [100, self.num_users]))
            s_zero_sub = tf.sparse.to_dense(tf.sparse_slice(self.S_zero, [self.segment_u,0], [100, self.num_users]))
            loss_edge += -tf.reduce_sum(tf.multiply(s_one_sub,tf.log(all_channel_possibility+eps))
                                        + tf.multiply(s_zero_sub,tf.log(1-all_channel_possibility+eps))) / (100 * self.num_users)
            
            y_homo_sub = tf.sparse.to_dense(tf.sparse_slice(self.Y_homo, [self.segment_u,0], [100, self.num_users]))
            y_hetero_sub = tf.sparse.to_dense(tf.sparse_slice(self.Y_hetero, [self.segment_u,0], [100, self.num_users]))
            loss_homo += -tf.reduce_sum(tf.multiply(tf.cast(y_homo_sub>0, tf.float32), tf.log(all_homo_possibility+eps)) 
                                        + tf.multiply(tf.cast(tf.logical_not(y_homo_sub>0), tf.float32), tf.log(1-all_homo_possibility+eps))) / (100 * self.num_users) 
            loss_hetero += -tf.reduce_sum(tf.multiply(tf.cast(y_hetero_sub>0, tf.float32), tf.log(all_hetero_possibility+eps)) 
                                          + tf.multiply(tf.cast(tf.logical_not(y_hetero_sub>0), tf.float32), tf.log(1-all_hetero_possibility+eps))) / (100 * self.num_users) 
                
        user_embeddings = tf.reduce_sum(all_user_embeddings, axis=0)
        item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0)
        
        #Prediction
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, item_embeddings), 1)
        
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
            
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding)) 
        loss_joint = loss_edge + 10 * (loss_homo + loss_hetero)
        
        opt = tf.train.AdamOptimizer(self.lRate)
        train_pre = opt.minimize(loss, var_list=[self.r_weights, self.weights])
        train = opt.minimize(loss+loss_joint, var_list=[self.r_weights, self.weights])
        train_d = opt.minimize(loss_channel, var_list=[self.weights, self.d_weights])
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Pretrain for consistan weights')
        for epoch in range(self.maxEpoch // 2):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                i_i = self.positive_sample(u_i)
                _, l, self.U, self.V = self.sess.run([train_pre, loss, user_embeddings, item_embeddings],
                                      feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.segment_u: u_i, self.positive_i: i_i, self.isOn: 0})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
            self.ranking_performance(self.maxEpoch // 2 + epoch)

        print('Train for hidden users')
        for epoch in range(self.maxEpoch // 2):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                i_i = self.positive_sample(u_i)
                _, _, l1, l2, l3, l4, self.U, self.V = self.sess.run([train, train_d, loss, loss_edge, loss_homo+loss_hetero, loss_channel, user_embeddings, item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.segment_u: u_i, self.positive_i: i_i, self.isOn: 1})
                print('training:', self.maxEpoch // 2 + epoch + 1, 'batch', n, 'loss1:', l1, 'loss2:', l2, 'loss3:', l3, 'loss4:', l4)
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
    
    
 
        
    