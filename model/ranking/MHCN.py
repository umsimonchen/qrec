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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）,here only FATAL

# Recommended Maximum Epoch Setting: LastFM 120 Douban 30 Yelp 30
# A slight performance drop is observed when we transplanted the model from python2 to python3. The cause is unclear.

class MHCN(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(MHCN, self).readConfiguration()
        args = config.OptionConf(self.config['MHCN'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]] # row index
            col += [self.data.user[pair[1]]] # column index
            entries += [1.0] # corresponding value of row and column in the same position 
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
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
        B = S.multiply(S.T) # csr_matrix point-wise multiplication
        U = S - B #lastFM has no element in U
        
        
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9+A9.T
        A10  = Y.dot(Y.T)-A8-A9
        
        #addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1)) # axis=0 -> column; axis=1 -> row. reshape -1 means unknown, 1 means 1 value
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>1) #reduce noise
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        
        T0 = (A8+A9!=0).astype(int)
        T0 = csr_matrix(T0)
        
        T1b = B.dot(B)
        T1r = B.dot(U) + U.dot(B) + U.dot(U)
        T1l = (B.dot(U) + U.dot(B) + U.dot(U)).T
        T1 = (Y.dot(Y.T)).multiply(T1b + T1r + T1l)
        '''
        T1.setdiag(0)
        T1.eliminate_zeros()
        T1 = (T1!=0).astype(int)
        T1 = csr_matrix(T1)
        T1 = (T0==0).multiply(T1)
        
        T2b = T1b.dot(B)
        T2r = T1r.dot(U) + U.dot(T1r)
        T2l = (T1r.dot(U) + U.dot(T1r)).T
        T2 = T2b + T2r + T2l
        T2.setdiag(0)
        T2.eliminate_zeros()
        T2 = (T2!=0).astype(int)
        T2 = csr_matrix(T2)
        T2 = ((T0==0).multiply(T1==0)).multiply(T2)
        '''
        D0 = (S!=0).astype(int)
        D0 = csr_matrix(D0)
        
        D1 = D0.dot(D0)
        D1.setdiag(0)
        D1.eliminate_zeros()
        D1 = (D1!=0).astype(int)
        D1 = csr_matrix(D1)
        D1 = (D0==0).multiply(D1)
        
        D2 = D1.dot(D0)
        D2.setdiag(0)
        D2.eliminate_zeros()
        D2 = (D1!=0).astype(int)
        D2 = csr_matrix(D2)
        D2 = ((D0==0).multiply(D1==0)).multiply(D2)
        
        #T = T2*1 +T1*2 + T0*3
        T = T1.multiply(1.0/T1.sum(axis=1).reshape(-1, 1))
        #T.setdiag(1)
        D = D2*1 +D1*2 + D0*3
        D = D.multiply(1.0/D.sum(axis=1).reshape(-1, 1))
        D.setdiag(1)

        '''
        A1 = (Y.dot(Y.T)).multiply(B)
        A2 = (Y.dot(Y.T)).multiply(U)
        A2 = A2+A2.T
        A3 = Y.dot(Y.T)-A1-A2
        
        T1b = B.dot(B)
        T1r = B.dot(U) + U.dot(B) + U.dot(U)
        T1l = (B.dot(U) + U.dot(B) + U.dot(U)).T
        T1 = T1b + T1r + T1l
        T1.setdiag(0)
        T1.eliminate_zeros()
        A4 = (Y.dot(Y.T)).multiply(T1)
        A4 = (((A1+A2) == 0).astype(int)).multiply(A4)
        
        T2b = T1b.dot(B)
        T2r = T1r.dot(U) + U.dot(T1r)
        T2l = (T1r.dot(U) + U.dot(T1r)).T
        T2 = T2b + T2r + T2l
        T2.setdiag(0)
        T2.eliminate_zeros()
        A5 = (Y.dot(Y.T)).multiply(T2)
        A5 = (((A1+A2+A4) == 0).astype(int)).multiply(A5)
        
        H_s = sum(1/2*A4,1/4*A5)
        H_s = H_s.multiply(H_s>1)
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1)) # axis=0 -> column; axis=1 -> row. reshape -1 means unknown, 1 means 1 value
        H_j = sum([A1,A2])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A3
        H_p = H_p.multiply(H_p>1) #reduce noise
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        print(A3.max,A4.max,A5.max,H_p.max)  
        '''
        return [H_s,H_j,H_p,T,D]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def initModel(self):
        super(MHCN, self).initModel() #deep-iterative
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
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at') #inter attention
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm') #intra attention
        #define inline functions
        def self_gating(em,channel):
            #broadcasting: m*d
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel])) 
        def self_supervised_gating(em, channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % channel])+self.weights['sgating_bias%d' % channel]))
        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                #m*d->m*1->3*m
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])),1)) 
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
        T = M_matrices[3]
        T = self.adj_to_sparse_tensor(T)
        D = M_matrices[4]
        D = self.adj_to_sparse_tensor(D)
        R = self.buildJointAdjacency() #build heterogeneous graph
        #self-gating
        user_embeddings_c1 = self_gating(self.user_embeddings, 1)
        user_embeddings_c2 = self_gating(self.user_embeddings, 2)
        user_embeddings_c3 = self_gating(self.user_embeddings, 3)
        simple_user_embeddings = self_gating(self.user_embeddings,4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]

        self.ss_loss = 0 #self-supervised loss
        #multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
            #Channel S
            #user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s,user_embeddings_c1)
            #norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1) #normalize the user embedding in differet dimension of a user
            #all_embeddings_c1 += [norm_embeddings]
            #Channel J
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]
            #Channel P
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            #Channel T
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(T, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            #Channel D
            #user_embeddings_c4 = tf.sparse_tensor_dense_matmul(D, user_embeddings_c4)
            #norm_embeddings = tf.math.l2_normalize(user_embeddings_c4, axis=1)
            #all_embeddings_c4 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        #averaging the channel-specific embeddings 
        #use reduce_sum rather than reduce_mean to amplify the embedding, accel the training
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        #aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3)
        self.final_user_embeddings += simple_user_embeddings/2
        #create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,1), H_j)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,2), H_p)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,3), T)
        #self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,4), D)
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
                _, l1, l2 = self.sess.run([train_op, rec_loss, self.ss_loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1,)#,'ss_loss',l2
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings]) #after sess.run() tensor turn into array
            self.ranking_performance(epoch) #iterative
    #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u]) #sum product over D: n*d . 1*d = n
        else:
            return [self.data.globalMean] * self.num_items #assume all are interested