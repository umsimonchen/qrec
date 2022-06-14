# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:51:47 2022

@author: simon
"""
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from util.loss import bpr_loss
import os
from util import config
from math import sqrt
from tqdm import tqdm
from random import randrange, choice
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.reset_default_graph()

class SLDR(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
    
    def readConfiguration(self):
        super(SLDR, self).readConfiguration()
        args = config.OptionConf(self.config['SLDR'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # asymmetric matrix - directed graph
            row += [self.data.user[pair[0]]] # row index
            col += [self.data.user[pair[1]]] # column index
            entries += [1.0] # corresponding value of row and column in the same position 
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # asymmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData] #change the item ID to order index
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData] # 1 / sqrt(#item the user connect) / sqrt(#user the item connect)
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr() # user as row -- csr: compressed sparse row: save the value at the same place once
        self.itemAdjacency = Y.T.tocsr() # item as row
        Bi = S.multiply(S.T) # csr_matrix point-wise multiplication
        Uni = S - Bi #lastFM has no element in Uni
        C1 = (Uni.dot(Uni)).multiply(Uni.T)
        A1 = C1 + C1.T
        C2 = (Bi.dot(Uni)).multiply(Uni.T) + (Uni.dot(Bi)).multiply(Uni.T) + (Uni.dot(Uni)).multiply(Bi)
        A2 = C2 + C2.T
        C3 = (Bi.dot(Bi)).multiply(Uni) + (Bi.dot(Uni)).multiply(Bi) + (Uni.dot(Bi)).multiply(Bi)
        A3 = C3 + C3.T
        A4 = (Bi.dot(Bi)).multiply(Bi)
        C5 = (Uni.dot(Uni)).multiply(Uni) + (Uni.dot(Uni.T)).multiply(Uni) + (Uni.T.dot(Uni)).multiply(Uni)
        A5 = C5 + C5.T
        A6 = (Uni.dot(Bi)).multiply(Uni) + (Bi.dot(Uni.T)).multiply(Uni.T) + (Uni.T.dot(Uni)).multiply(Bi)
        A7 = (Uni.T.dot(Bi)).multiply(Uni.T) + (Bi.dot(Uni)).multiply(Uni) + (Uni.dot(Uni.T)).multiply(Bi)
        A8 = (Y.dot(Y.T)).multiply(Bi)
        A9 = (Y.dot(Y.T)).multiply(Uni)
        A9 = A9+A9.T
        A10  = Y.dot(Y.T)-A8-A9
        
        #addition and row-normalization
        H = sum([A1,A2,A3,A4,A5,A6,A7,A8,A9,A10])
        H = H.multiply(1.0/H.sum(axis=1).reshape(-1, 1)) # axis=0 -> column; axis=1 -> row. reshape -1 means unknown, 1 means 1 value

        return [H]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def initModel(self):
        super(SLDR, self).initModel()
        #print(self.data.trainSet_i)
        self.pos_i = tf.placeholder(tf.int32, shape=[self.num_users])
        self.segment_u = tf.placeholder(tf.int32)
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0) #(m+n)*d
        norm_adj = self.create_joint_sparse_adj_tensor()
        all_new_embeddings = []
        for k in range(self.n_layers):
            new_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            all_new_embeddings.append(new_embeddings)
        
        n_embeddings = tf.reduce_sum(all_new_embeddings, axis=0)
        self.final_user_embeddings, self.final_item_embeddings = tf.split(n_embeddings, [self.num_users, self.num_items], axis=0)
        #embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx) #placeholder in deepRecommender.py
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
    
    def trainModel(self):
        # no super class
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.final_user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss+reg_loss #+ self.ss_rate*self.ss_loss_u
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer() #when tf.Variable exists 
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch #size of each of them are batch_size
                _, l1, self.U, self.V = self.sess.run([train_op, rec_loss, self.final_user_embeddings, self.item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1)#,'ss_loss',l2
            self.ranking_performance(epoch) #iterative
        self.U,self.V = self.bestU,self.bestV


    def saveModel(self):
        self.bestU, self.bestV = self.U, self.V

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u]) #sum product over D: n*d . 1*d = n
        else:
            return [self.data.globalMean] * self.num_items #assume all are interested
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    