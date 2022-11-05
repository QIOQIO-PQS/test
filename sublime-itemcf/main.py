import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model import GCN, GCL
from utils import *
from sklearn.cluster import KMeans
from lightgcn import *
from dataloader import *
import random

EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu


    def loss_gcl(self, model, graph_learner, anchor_adj):
        features = graph_learner.embedding_item.weight
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner()
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj


    def cf_test(self, X, k):
        rec_items = X[0].numpy()
        groundtrue_items = X[1]
        r = getLabel(groundtrue_items,rec_items)
        
        recall = Recall_ATk(groundtrue_items, r, k)
        ndcg = NDCGatK_r(groundtrue_items, r, k )
        return {'recall':recall,
                'ndcg':ndcg}
        

    def get_rating(self, rating, itemAdj):
    #    return np.matmul(rating,itemAdj)
        return torch.matmul(rating,itemAdj.double())


    def itemCF(self, dataset, itemAdj, rating, k):
        max_K = k
        testDict = dataset.testDict
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        results = {'recall':0.0,'ndcg':0.0}
        allPos = dataset.getUserPosItems(users)
        groundTrue = [testDict[u] for u in users]
        batch_users_gpu = torch.Tensor(users).long()
        batch_users_gpu = batch_users_gpu.to(torch.device('cuda'))

        rating = self.get_rating(rating, itemAdj)

        #rating = rating.cpu()
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        
        del rating
        users_list.append(users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)

        # test rating_K with groundTrue 
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(self.cf_test(x,k))

        for result in pre_results:
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        return results['recall'], results['ndcg']
    
    def train(self, args):
        dataset = Dataset(args.dataset)
        torch.cuda.set_device(args.gpu)
        rating = dataset.rating.todense()
        rating = torch.from_numpy(rating)
        adj_original = dataset.ItemGraph
        
        for trial in range(args.ntrials):
            self.setup_seed(trial)
            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(dataset.n_item)
                else:
                    anchor_adj_raw = torch.eye(dataset.n_item)
            elif args.gsl_mode == 'structure_refinement':   
                anchor_adj_raw = torch.from_numpy(adj_original)
            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            graph_learner = LightGCN_learner(n_layers=2, isize=args.in_dim, k=args.k, 
                              knn_metric=args.sim_function, dataset=dataset, i=6)
            
            model = GCL(nlayers=args.nlayers, in_dim=args.in_dim, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)


            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()  
                anchor_adj = anchor_adj.cuda()
                rating = rating.cuda()
            
            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                loss, Adj = self.loss_gcl(model, graph_learner, anchor_adj)
                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)
                f_adj = Adj.detach()
                new_rating = self.get_rating(rating,f_adj)
                loss += -0.001*(F.log_softmax(new_rating,-1)*rating).sum(1).mean()

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                

                print("Epoch {:04d} | CL Loss {:.4f}".format(epoch, loss.item()))
                
                best_recall, best_ndcg, best_epoch = 0, 0, 0
                if epoch % args.eval_freq == 0:
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj

                    f_adj = f_adj.detach()

                    recall, ndcg = self.itemCF(dataset, f_adj , rating , args.evalk)
                    self.print_results(recall, ndcg, epoch, args)

                    if recall>best_recall and ndcg>best_ndcg:
                        best_recall = recall
                        best_ndcg = ndcg
                        best_epoch = epoch
            print("trial{:02d} best results:".format(trial))
            self.print_results(best_recall, best_ndcg, best_epoch, args)

    def print_results(self, recall, ndcg, epoch, args):
        print("Epoch:{0:04d}/{1}  Recall@{2}:{3:.4f} ndcg@{2}:{4:.4f}".format(epoch-1, args.epochs, args.evalk, recall, ndcg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='ml-1m',
                        choices=['Gowalla', 'Epinion', 'Ciao','ml-1m'])
    parser.add_argument('-ntrials', type=int, default=1)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_refinement",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=50)
    '''
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    '''                    
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.05)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-in_dim', type=int, default=128)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    # parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=50)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation ItemCF
    parser.add_argument('-evalk', type=int, default=10)
    '''
    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)
    '''
    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
