import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import multiprocessing as mp
from collections import defaultdict
from scipy.sparse import csr_matrix

class Dataset():
    def __init__(self, dataset) -> None:
        print("loading dataset")
        trainData = pd.read_csv('./data/'+dataset+'/train.csv')
        testData  = pd.read_csv('./data/'+dataset+'/test.csv')

        self.n_user = max(max(trainData.UserID.to_numpy()), max(testData.UserID.to_numpy()))+1
        self.n_user = self.n_user.item()
        self.n_item = max(max(trainData.ItemID.to_numpy()), max(testData.ItemID.to_numpy()))+1
        self.n_item = self.n_item.item()

        self.trainuser = np.array(trainData['UserID'])
        self.trainitem = np.array(trainData['ItemID'])
        self.rating = csr_matrix((np.ones(len(self.trainuser)),(self.trainuser,self.trainitem)), \
                                    shape=(self.n_user,self.n_item))
        self.ItemGraph = self.getItemGraph()
        #testData
        self.testuser = np.array(testData['UserID'])
        self.testitem = np.array(testData['ItemID'])
        self.Graph = None

        self.testDict = self.build_test()
        
    def getSparseGraph(self):
        if self.Graph is None:
            graphuser = torch.LongTensor(self.trainuser)
            graphitem = torch.LongTensor(self.trainitem)
            subgraph1 = torch.stack([graphuser.long(),graphitem.long()+self.n_user])
            subgraph2 = torch.stack([graphitem.long()+self.n_user,graphuser.long()])
            index = torch.cat([subgraph1,subgraph2],dim=1)
            data = torch.ones(2*len(self.trainuser))
            self.Graph = torch.sparse.FloatTensor(index,data,torch.Size([self.n_user+self.n_item,self.n_user+self.n_item]))
            
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_user+self.n_item, self.n_user+self.n_item]))
        return self.Graph 
        
    def getItemGraph(self):
        '''
        get item graph by rating
        return torch.tensor
        '''
        #dense matrix multiply, too slow
        #rating = self.rating.todense()
        #ItemMatrix = np.matmul(rating.T,rating)

        #sparse matrix multiply
        item_sparse = self.rating.transpose().dot(self.rating)
        return item_sparse.todense()
         
    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testitem):
            user = self.testuser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []  
        for user in users:
            posItems.append(self.rating[user].nonzero()[1])
        return posItems

    def load_rating(self,dataset):
        df = pd.read_csv(os.path.join('./data/'+dataset+'/train.csv'))
        user_list = df.UserID.to_numpy()
        item_list = df.ItemID.to_numpy()
        rating_list = df.Rating.to_numpy()
        items_user_interacted = defaultdict(list)
        ratings_user_rating = defaultdict(list)
        users_item_interacted = defaultdict(list)
        ratings_item_rated = defaultdict(list)
        for idx in range(user_list.shape[0]):
            user_id = user_list[idx]
            item_id = item_list[idx]
            rating = rating_list[idx]
            items_user_interacted[user_id].append(item_id)
            ratings_user_rating[user_id].append(rating)
            users_item_interacted[item_id].append(user_id)
            ratings_item_rated[item_id].append(rating) 
        return items_user_interacted,ratings_user_rating 
