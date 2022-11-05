import torch
import torch.nn as nn
from dataloader import Dataset
from utils import *


class LightGCN_learner(nn.Module):
    def __init__(self, n_layers, isize, k, knn_metric, i, dataset):
        super(LightGCN_learner, self).__init__()

        self.dataset = dataset
        self.Graph = self.dataset.getSparseGraph()
        self.layers = nn.ModuleList()
        
        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric

        self.i = i
        self.act = nn.Sigmoid()
        self.non_linearity = 'relu'
        
        self.n_layers = n_layers
        self.num_users = self.dataset.n_user
        self.num_items = self.dataset.n_item
        self.embedding_user = nn.Embedding(num_embeddings = self.num_users , 
            embedding_dim = self.input_dim)
        self.embedding_item = nn.Embedding(num_embeddings = self.num_items , 
            embedding_dim = self.input_dim)
        self.param_init()
        

    
    def internal_forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.Graph.cuda()   
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items


    def param_init(self):
        nn.init.normal_(self.embedding_user.weight , std=0.1)
        nn.init.normal_(self.embedding_item.weight , std=0.1)


    def forward(self):
        _, embeddings = self.internal_forward()
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        similarities = top_k(similarities, self.k + 1)
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities