from base.iterativeRecommender import IterativeRecommender
from random import shuffle,randint,choice
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()

    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U') # m*d
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') # n*d
        self.batch_user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.u_idx) # get the user embedding of users with certain index
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx) # get the item embedding of items with certain index
        # dynamically apply graphic memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)] # real user id
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)] # real item id
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size
            
            u_idx, i_idx, j_idx = [], [], []
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

    def next_batch_pointwise(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size
            u_idx,i_idx,y = [],[],[]
            for i,user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
                y.append(1)
                for instance in range(4):
                    item_j = randint(0, self.num_items - 1)
                    while self.data.id2item[item_j] in self.data.trainSet_u[user]:
                        item_j = randint(0, self.num_items - 1)
                    u_idx.append(self.data.user[user])
                    i_idx.append(item_j)
                    y.append(0)
            yield u_idx,i_idx,y

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        pass


