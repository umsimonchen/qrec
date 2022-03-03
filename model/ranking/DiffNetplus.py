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
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#For general comparison. We do not include the user/item features extracted from text/images

class DiffNetplus(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(DiffNetplus, self).readConfiguration()
        args = config.OptionConf(self.config['DiffNetplus'])
        self.n_layers = int(args['-n_layer']) #the number of layers of the recommendation module (discriminator)

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
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.01), name='U') # m*d
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.01), name='V') # n*d
        S = self.buildSparseRelationMatrix()
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape) #social: m*m
        self.A = self.create_sparse_adj_tensor() #item: m*n, normalized

    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_fusion_weights'] = tf.Variable(
            initializer([self.emb_size, self.emb_size]), name='user_fusion_weights')
        self.weights['item_fusion_weights'] = tf.Variable(
            initializer([self.emb_size, self.emb_size]), name='item_fusion_weights')      
        user_embeddings = tf.matmul(self.user_embeddings,self.weights['user_fusion_weights']) #m*d
        item_embeddings = tf.matmul(self.item_embeddings,self.weights['item_fusion_weights']) #m*d
        all_user_embeddings = [user_embeddings]
        all_item_embeddings = [item_embeddings]
        
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
            i=1       
            new_item_embeddings = tf.identity(all_item_embeddings[k])
            self.item_list = tf.placeholder(tf.int32, shape=[100,])
            item_label = tf.constant(list(self.data.id2item.values()))
            buyers = [] 
            length = []
            for item in self.data.trainSet_i.keys():
                tmp = [0] * self.num_users
                for buyer in self.data.trainSet_i[item].keys():
                    tmp[self.data.user[buyer]] = 1
                length.append(sum(tmp))
                buyers.append(tmp)
            buyers = tf.constant(buyers)
            length = tf.constant(length)
            users = tf.constant([i+1 for i in range(self.num_users)])
            
            for i in range(100):
                #item_index = (i_Segment + i) % self.num_items
                item_embedding = tf.nn.embedding_lookup(all_item_embeddings[k], self.v_idx)
                item = tf.gather(item_label, self.v_idx)
                #buyers_index = tf.multiply(tf.gather(buyers, i), users) - 1
                #for user in self.data.trainSet_i[item].keys():
                    #buyers_index.append(self.data.user[user])
                #buyers_embedding = tf.gather(all_user_embeddings[k], buyers_index)
                vu_ego_embedding = tf.concat([[item_embedding]*self.num_users, all_user_embeddings[k]], 1)
                vu_ego_embedding = tf.matmul(vu_ego_embedding,self.weights['mlp_one_in_weights%d' % k])
                vu_ego_embedding = tf.matmul(vu_ego_embedding,self.weights['mlp_one_out_weights%d' % k])
                #vu_ego_embedding = tf.gather(vu_ego_embedding,buyers_index)
                vu_ego_embedding = tf.nn.relu(vu_ego_embedding)
                #l = tf.gather(length, item_index)
                eta = tf.nn.softmax(tf.reshape(vu_ego_embedding, [1, self.num_users]))
                
                new_item_embedding =  tf.matmul(eta,all_user_embeddings[k])
                tmp = tf.concat([new_item_embeddings[:self.v_idx], new_item_embedding], 0)
                new_item_embeddings = tf.concat([tmp, new_item_embeddings[self.v_idx+1:]], 0)
                print("Finished MLP 1: %d/%d/Layer %d" %(i+1,100,k+1))
                #i+=1
            
            #User diffusion
            i=1
            new_user_embeddings = tf.identity(all_user_embeddings[k])
            '''
            for user in self.social.followees.keys():
                user_index = self.data.user[user]
                user_embedding = tf.gather(all_user_embeddings[k], user_index)
                
                #User neighbor
                followees_index = []
                for followee in self.social.followees[user].keys():
                    followees_index.append(self.data.user[followee])
                followees_embedding = tf.gather(all_user_embeddings[k], followees_index)
                uu_ego_embedding = tf.concat([[user_embedding]*len(self.social.followees[user]), followees_embedding], 1)
                uu_ego_embedding = tf.matmul(uu_ego_embedding,self.weights['mlp_two_in_weights%d' % k])
                uu_ego_embedding = tf.matmul(uu_ego_embedding,self.weights['mlp_two_out_weights%d' % k])
                uu_ego_embedding = tf.nn.relu(uu_ego_embedding)
                alpha = tf.nn.softmax(tf.reshape(uu_ego_embedding, [1, len(self.social.followees[user])]))
                
                user_neighbors_embedding = tf.matmul(alpha,followees_embedding)
                
                #Item neighbor
                items_index = []
                for item in self.data.trainSet_u[user].keys():
                    items_index.append(self.data.item[item])
                items_embedding = tf.gather(all_item_embeddings[k], items_index)
                uv_ego_embedding = tf.concat([[user_embedding]*len(self.data.trainSet_u[user]), items_embedding], 1)
                uv_ego_embedding = tf.matmul(uv_ego_embedding,self.weights['mlp_three_in_weights%d' % k])
                uv_ego_embedding = tf.matmul(uv_ego_embedding,self.weights['mlp_three_out_weights%d' % k])
                uv_ego_embedding = tf.nn.relu(uv_ego_embedding)
                beta = tf.nn.softmax(tf.reshape(uv_ego_embedding, [1, len(self.data.trainSet_u[user])]))
                
                item_neighbors_embedding = tf.matmul(beta,items_embedding)
                
                #Aggregation
                up_ego_embedding = tf.concat([[user_embedding], user_neighbors_embedding], 1)
                uq_ego_embedding = tf.concat([[user_embedding], item_neighbors_embedding], 1)
                upq_ego_embedding = tf.concat([up_ego_embedding, uq_ego_embedding], 0)
                upq_ego_embedding = tf.matmul(upq_ego_embedding,self.weights['mlp_four_in_weights%d' % k])
                upq_ego_embedding = tf.matmul(upq_ego_embedding,self.weights['mlp_four_out_weights%d' % k])
                upq_ego_embedding = tf.nn.relu(upq_ego_embedding)
                gamma = tf.nn.softmax(tf.reshape(upq_ego_embedding, [1, 2]))
                
                new_user_embedding = tf.matmul(gamma,tf.concat([user_neighbors_embedding, item_neighbors_embedding], 0))
                tmp = tf.concat([new_user_embeddings[:user_index], new_user_embedding], 0)                
                new_user_embeddings = tf.concat([tmp, new_user_embeddings[user_index+1:]], 0) 
                
                print("Finished MLP 2,3,4: %d/%d/Layer %d" %(i,len(self.social.followees),k+1))
                i+=1
            '''
            all_item_embeddings.append(new_item_embeddings)
            all_user_embeddings.append(new_user_embeddings)
        
        #Organization
        for k in range(1,self.n_layers):
            user_embeddings = tf.concat([user_embeddings, all_user_embeddings[k]], 1)
            item_embeddings = tf.concat([item_embeddings, all_item_embeddings[k]], 1)
            
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
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                #print(i_idx)
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx[0]})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u})
        else:
            return [self.data.globalMean] * self.num_items