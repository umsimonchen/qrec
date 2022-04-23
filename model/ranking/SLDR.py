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
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1)) # axis=0 -> column; axis=1 -> row. reshape -1 means unknown, 1 means 1 value
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>=1) #reduce noise
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        return [H_s,H_j,H_p]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def initModel(self):
        super(SLDR, self).initModel()
        #print(self.data.trainSet_i)
        self.pos_i = tf.placeholder(tf.int32, shape=[self.num_users])
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 4
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
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]
        
        self.weights['query_weights_ui'] = tf.Variable(
            initializer([2 * self.emb_size, self.emb_size]), name='query_weights_ui')
        def neighborWeight_ui(ego):
            index, u, pos = tf.split(tf.reshape(ego, [1, self.num_users+2*self.emb_size]), [self.num_users, self.emb_size, self.emb_size], axis=1)
            neighbors_concat = tf.concat([user_embeddings_c1, tf.tile(pos, [self.num_users, 1])], axis=1) #m*2d
            query = tf.nn.relu(tf.matmul(neighbors_concat, self.weights['query_weights_ui']))#m*d
            score = tf.math.square(tf.norm(query - tf.tile(u, [self.num_users, 1]), axis=1))#m
            score = tf.math.exp(-score)#m
            score = score / (tf.norm(score,1)+1e-8)
            return score
        
        dense_hp = tf.sparse_tensor_to_dense(H_p)
        neighbors_pos = tf.nn.embedding_lookup(item_embeddings, self.pos_i)#m*d
        neighbors_ego_hp = tf.concat([dense_hp, user_embeddings_c3, neighbors_pos], axis=1)#m*(m+d+d)
        neighbor_weight = tf.vectorized_map(fn=lambda em: neighborWeight_ui(em),elems=neighbors_ego_hp)
        dense_hp = tf.multiply(neighbor_weight, dense_hp)
        dense_hp = dense_hp / tf.reshape(tf.reduce_sum(dense_hp, axis=1), (-1, 1))
        arr_idx = tf.where(tf.not_equal(dense_hp, 0))
        H_p = tf.SparseTensor(arr_idx, tf.gather_nd(dense_hp, arr_idx), dense_hp.get_shape())
        self.neighbors_weight = neighbor_weight
        self.neighbors = dense_hp
        
        # self.weights['query_weights_uu'] = tf.Variable(
        #     initializer([2 * self.emb_size, self.emb_size]), name='query_weights_uu')
        # def neighborWeight_uu(ego):
        #     index, u, pos = tf.split(tf.reshape(ego, [1, self.num_users+2*self.emb_size]), [self.num_users, self.emb_size, self.emb_size], axis=1)
        #     neighbors_concat = tf.concat([user_embeddings_c1, tf.tile(pos, [self.num_users, 1])], axis=1) #m*2d
        #     query = tf.nn.relu(tf.matmul(neighbors_concat, self.weights['query_weights_uu']))#m*d
        #     score = tf.math.square(tf.norm(query - tf.tile(u, [self.num_users, 1]), axis=1))#m
        #     score = tf.math.exp(-score)#m
        #     score = score / (tf.norm(score,1)+1e-8)
        #     return score
        
        # dense_hs = tf.sparse_tensor_to_dense(H_s)
        # neighbors_ego_hs = tf.concat([dense_hs, user_embeddings_c1, neighbors_pos], axis=1)
        # neighbor_weight_hs = tf.vectorized_map(fn=lambda em: neighborWeight_uu(em), elems)
        
        self.ss_loss_u = 0 #self-supervised loss
        #multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding_user = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3, t='u')[0] + simple_user_embeddings / 2
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
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding_user)
            all_embeddings_i += [tf.math.l2_normalize(new_item_embeddings, axis=1)]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple_user += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings

        #averaging the channel-specific embeddings - why reduce_sum?????????????????????
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple_user, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        #aggregating channel-specific embeddings
        self.final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3,t='u')
        self.final_user_embeddings += simple_user_embeddings/2
        self.final_item_embeddings = item_embeddings
        #create self-supervised loss
        self.ss_loss_u += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,1), H_s)
        self.ss_loss_u += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,2), H_j)
        self.ss_loss_u += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,3), H_p)
        #embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx) #placeholder in deepRecommender.py
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
    
    def sample(self):
        pos_i = []
        for i in range(self.num_users):
            items = self.data.trainSet_u[self.data.id2user[i]]
            item = choice(list(items.items()))[0]
            pos_i.append(self.data.item[item])
        return pos_i
    
    def trainModel(self):
        # no super class
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss+reg_loss #+ self.ss_rate*self.ss_loss_u
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer() #when tf.Variable exists 
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch #size of each of them are batch_size
                pos_i = self.sample()
                nei, nei1, _, l1, l2_u, self.U, self.V = self.sess.run([self.neighbors_weight, self.neighbors, train_op, rec_loss, self.ss_loss_u, self.final_user_embeddings, self.final_item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.pos_i: pos_i})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1, 'ss_loss_u:', l2_u)#,'ss_loss',l2
            #self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings]) #after sess.run() tensor turn into array
            self.ranking_performance(epoch) #iterative
    #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV
        import csv
        with open("out.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(nei)
        with open("out1.csv", "w", newline="") as f1:
            writer = csv.writer(f1)
            writer.writerows(nei1)

    def saveModel(self):
        self.bestU, self.bestV = self.U, self.V

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u]) #sum product over D: n*d . 1*d = n
        else:
            return [self.data.globalMean] * self.num_items #assume all are interested
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    