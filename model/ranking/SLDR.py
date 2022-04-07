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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    
    def buildSparseItemMatrix(self):
        row, col, entries = [], [], []
        for item1 in tqdm(self.data.trainSet_i.keys()):
            i1 = self.data.trainSet_i[item1]
            for item2 in self.data.trainSet_i.keys():
                if item1 != item2:
                    i2 = self.data.trainSet_i[item2]
                    inter = set(i1) & set(i2)
                    if (2*len(inter) > len(i1)) and (2*len(inter) > len(i2)):
                        row += [self.data.item[item1]]
                        col += [self.data.item[item2]]
                        entries += [1.0]
        itemMatrix = coo_matrix((entries, (row, col)), shape=(self.num_items,self.num_items),dtype=np.float32)
        #print(itemMatrix.nonzero())
        return itemMatrix
    
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
        C = self.buildSparseItemMatrix()
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
        
        B1 = (C.dot(C)).multiply(C)
        B2 = (Y.T.dot(Y)).multiply(C)
        B3 = Y.T.dot(Y)-B2
        #addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1)) # axis=0 -> column; axis=1 -> row. reshape -1 means unknown, 1 means 1 value
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>1) #reduce noise
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        H_i1 = B1
        H_i1 = H_i1.multiply(1.0/H_i1.sum(axis=1).reshape(-1, 1))
        H_i2 = B2
        H_i2 = H_i2.multiply(1.0/H_i2.sum(axis=1).reshape(-1, 1))
        H_i3 = B3
        print(H_i3.count_nonzero())
        H_i3 = H_i3.multiply(H_i3>1)
        print(H_i3.count_nonzero())
        H_i3 = H_i3.multiply(1.0/H_i3.sum(axis=1).reshape(-1, 1))
        return [H_s,H_j,H_p,H_i1,H_i2,H_i3]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def initModel(self):
        super(SLDR, self).initModel()
        #print(self.data.trainSet_i)
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 8
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        #define learnable paramters
        for i in range(self.n_channel):
            #base user embedding gating
            self.weights['gating%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1)) #d*d
            self.weights['gating_bias%d' %(i+1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1)) #d*1
            #self-supervised gating
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention_u'] = tf.Variable(initializer([1, self.emb_size]), name='at_u') #inter attention
        self.weights['attention_mat_u'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm_u') #intra attention
        self.weights['attention_i'] = tf.Variable(initializer([1, self.emb_size]), name='at_i') #inter attention
        self.weights['attention_mat_i'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm_i') #intra attention
        #define inline functions
        def self_gating(em,channel):
            #broadcasting: m*d
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel])) 
        def self_supervised_gating(em, channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % channel])+self.weights['sgating_bias%d' % channel]))
        def channel_attention(*channel_embeddings, t):
            weights = []
            for embedding in channel_embeddings:
                #m*d->m*1->3*m
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention_'+t], tf.matmul(embedding, self.weights['attention_mat_'+t])),1)) 
            #every user has a attention score for every channel, m*3; default softmax axis=-1(last dimension);
            score = tf.nn.softmax(tf.transpose(weights)) 
            mixed_embeddings = 0 #broadcasting
            for i in range(len(weights)):
                #1*m⊙d*m: broadcasting
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings,score
        #initialize adjacency matrices
        H_s = M_matrices[0]
        H_s = self.adj_to_sparse_tensor(H_s) #turn sparse matrix into tensor
        H_j = M_matrices[1]
        H_j = self.adj_to_sparse_tensor(H_j)
        H_p = M_matrices[2]
        H_p = self.adj_to_sparse_tensor(H_p)
        H_i1 = M_matrices[3]
        H_i1 = self.adj_to_sparse_tensor(H_i1)
        H_i2 = M_matrices[4]
        H_i2 = self.adj_to_sparse_tensor(H_i2)
        H_i3 = M_matrices[5]
        H_i3 = self.adj_to_sparse_tensor(H_i3)
        R = self.buildJointAdjacency() #build heterogeneous graph
        #self-gating
        user_embeddings_c1 = self_gating(self.user_embeddings,1)
        user_embeddings_c2 = self_gating(self.user_embeddings, 2)
        user_embeddings_c3 = self_gating(self.user_embeddings, 3)
        simple_user_embeddings = self_gating(self.user_embeddings,4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple_user = [simple_user_embeddings]
        
        item_embeddings_d1 = self_gating(self.item_embeddings,5)
        item_embeddings_d2 = self_gating(self.item_embeddings,6)
        item_embeddings_d3 = self_gating(self.item_embeddings,7)
        simple_item_embeddings = self_gating(self.item_embeddings,8)
        all_embeddings_d1 = [item_embeddings_d1]
        all_embeddings_d2 = [item_embeddings_d2]
        all_embeddings_d3 = [item_embeddings_d3]
        all_embeddings_simple_item = [simple_item_embeddings]

        self.ss_loss = 0 #self-supervised loss
        #multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding_user = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3, t='u')[0] + simple_user_embeddings / 2
            mixed_embedding_item = channel_attention(item_embeddings_d1, item_embeddings_d2, item_embeddings_d3, t='i')[0] + simple_item_embeddings / 2
            #Channel S
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s,user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1) #normalize the user embedding in differet dimension of a user
            all_embeddings_c1 += [norm_embeddings]
            #Channel J
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            #Channel P
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            #Channel I1
            item_embeddings_d1 = tf.sparse_tensor_dense_matmul(H_i1, item_embeddings_d1)
            norm_embeddings = tf.math.l2_normalize(item_embeddings_d1, axis=1)
            all_embeddings_d1 += [norm_embeddings]
            #Channel I2
            item_embeddings_d2 = tf.sparse_tensor_dense_matmul(H_i2, item_embeddings_d2)
            norm_embeddings = tf.math.l2_normalize(item_embeddings_d2, axis=1)
            all_embeddings_d2 += [norm_embeddings]
            #Channel I3
            item_embeddings_d3 = tf.sparse_tensor_dense_matmul(H_i3, item_embeddings_d3)
            norm_embeddings = tf.math.l2_normalize(item_embeddings_d3, axis=1)
            all_embeddings_d3 += [norm_embeddings]
            # item convolution
            simple_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding_user)
            all_embeddings_simple_item += [tf.math.l2_normalize(simple_item_embeddings, axis=1)]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, mixed_embedding_item)
            all_embeddings_simple_user += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            #item_embeddings = new_item_embeddings
        #averaging the channel-specific embeddings - why reduce_sum?????????????????????
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple_user, axis=0)
        item_embeddings_d1 = tf.reduce_sum(all_embeddings_d1, axis=0)
        item_embeddings_d2 = tf.reduce_sum(all_embeddings_d2, axis=0)
        item_embeddings_d3 = tf.reduce_sum(all_embeddings_d3, axis=0)
        simple_item_embeddings = tf.reduce_sum(all_embeddings_simple_item, axis=0)
        #aggregating channel-specific embeddings
        self.final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3,t='u')
        self.final_item_embeddings,self.attention_score = channel_attention(item_embeddings_d1,item_embeddings_d2,item_embeddings_d3,t='i')
        self.final_user_embeddings += simple_user_embeddings/2
        self.final_item_embeddings += simple_item_embeddings/2
        #create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,1), H_s)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,2), H_j)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,3), H_p)
        #self-supervised loss for item should be less important than user - 0.5/0.1*ss_loss 
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_item_embeddings,5), H_i1)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_item_embeddings,6), H_i2)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_item_embeddings,7), H_i3)
        #embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
            
    def hierarchical_self_supervision(self,em,adj):
        def row_shuffle(embedding):
            #get the total size m -> enumerate m as index -> shuffle -> gather by index
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0]))) 
        def row_column_shuffle(embedding):
            #column shuffle
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            #row shuffle
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        user_embeddings = em
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj,user_embeddings) #sub-hypergraph representation m*d
        #Local MIM
        pos = score(user_embeddings,edge_embeddings) #m*d m*d -> m*d -> m 
        neg1 = score(row_shuffle(user_embeddings),edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings),user_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2))) #-tf.log(tf.sigmoid(neg1-neg2))
        #Global MIM
        graph = tf.reduce_mean(edge_embeddings,0)
        pos = score(edge_embeddings,graph)
        neg1 = score(row_column_shuffle(edge_embeddings),graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))) #-tf.log(tf.sigmoid(neg1-neg2))
        return global_loss+local_loss
        
    def trainModel(self):
        # no super class
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss+reg_loss + self.ss_rate*self.ss_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer() #when tf.Variable exists 
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1, l2, self.U, self.V = self.sess.run([train_op, rec_loss, self.ss_loss, self.final_user_embeddings, self.final_item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1, 'ss_loss', l2 )#,'ss_loss',l2
            #self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings]) #after sess.run() tensor turn into array
            self.ranking_performance(epoch) #iterative
    #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    