import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from model import HINormer
from utils.data import load_data
from utils.pytorchtools import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append('utils/')

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    labels = torch.FloatTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)

    all_nodes = np.arange(features_list[0].shape[0])

    node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()

    n = 0

    for x in all_nodes:

        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        while (cnt < args.len_seq):
            sample_list = g.successors(start).numpy().tolist()
            nsampled = max(len(sample_list), 1)
            sampled_list = random.sample(sample_list, nsampled)
            for i in range(nsampled):
                node_seq[n, cnt] = sampled_list[i]
                cnt += 1
                if cnt == args.len_seq:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()
        n += 1

    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]

    g = dgl.add_self_loop(g)
    g = g.to(device)
    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    num_classes = dl.labels_train['num_classes']
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)

    loss=torch.nn.BCELoss()

    for i in range(args.repeat):
        
        net = HINormer(g, num_classes, in_dims, args.hidden_dim, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
                    temper=args.temperature, num_type=len(node_cnt), beta = args.beta)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/HINormer_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            logits = net(features_list, train_seq, type_emb, node_type, args.l2norm)
            logp = logits.sigmoid()
            train_loss = loss(logp, labels[train_idx])

            # autograd
            optimizer.zero_grad() 
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()

            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, val_seq, type_emb, node_type, args.l2norm)
                logp=logits.sigmoid()
                val_loss = loss(logp, labels[val_idx])
                pred=(logits.cpu().numpy() > 0).astype(int)
                print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))
    
            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/HINormer_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device)))
        net.eval()
        with torch.no_grad():
            logits = net(features_list, test_seq, type_emb, node_type, args.l2norm)
            test_logits=logits
            pred=(test_logits.cpu().numpy() > 0).astype(int)
            if args.mode == 1:
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                result=dl.evaluate(pred)
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']
    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='HINormer')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=256,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--dataset', type=str, default = 'DBLP', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
    ap.add_argument('--num-gnns', type=int, default=4, help='The number of layers of both structural and heterogeneous encoder')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=200, help='The length of node sequence.')
    ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
    ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)


    args = ap.parse_args()
    run_model_DBLP(args)
