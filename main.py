import random
from time import time

import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from metrics import calc_metrics_at_k_
from parser import parse_args
from dataloader import DataLoader_LS, DataLoader_S, DataLoader_Si
from models import KPCR_LS, KPCR_S
from helper import early_stopping


def evaluate(model, train_user_dict, test_user_dict,
             user_ids_batches, item_ids, data, K):
    model.eval()
    n_users = len(test_user_dict.keys())
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []

    with torch.no_grad():
        for user_ids_batch in user_ids_batches:
            # (n_batch_users, n_eval_items)
            cf_scores_batch = model.predict(user_ids_batch, item_ids)
            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k_(
                cf_scores_batch, train_user_dict, test_user_dict,
                user_ids_batch, item_ids_batch, data.n_users, K)
            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)
        cf_scores = np.concatenate(cf_scores, axis=0)
        precision_k = sum(np.concatenate(precision)) / n_users
        recall_k = sum(np.concatenate(recall)) / n_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_users
        return cf_scores, precision_k, recall_k, ndcg_k


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if args.kpcr_mode == 'ls':
        data = DataLoader_LS(args)
        model = KPCR_LS(
            args, data.level_graph, data.n_users, data.n_items,
            data.n_users_entities, data.n_relations)
    elif args.kpcr_mode == 's':
        data = DataLoader_S(args)
        model = KPCR_S(
            args, data.n_users, data.n_items, data.n_users_entities,
            data.n_relations)
    else:
        data = DataLoader_Si(args)
        model = KPCR_S(
            args, data.n_users, data.n_items, data.n_users_entities,
            data.n_relations)
    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in
                        range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    item_ids = torch.arange(data.n_users, data.n_users+data.n_items,
                            dtype=torch.long)
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]
        item_ids = item_ids.to(device)
        model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []
    history = []
    if args.kpcr_mode == 'ls':
        lv_history = []
        model.level_graph = dgl.remove_self_loop(model.level_graph)
        model.level_graph = dgl.add_self_loop(model.level_graph)
        degs = model.level_graph.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        model.level_graph.ndata['norm'] = norm.unsqueeze(1)
        model.level_graph = model.level_graph.to('cpu')
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            model.level_graph, model.level_graph.nodes(), sampler,
            batch_size=2048, shuffle=True, drop_last=False,
            num_workers=2)

    for epoch in range(1, args.n_epoch):
        model.train()
        if args.kpcr_mode == 'ls':
            time1 = time()
            model.level_graph = model.level_graph.to('cpu')
            lv_loss_func = nn.BCEWithLogitsLoss()
            for _, _, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                features = blocks[0].srcdata['feat']
                labels = blocks[-1].dstdata['level'].type(
                    torch.FloatTensor).to(device)
                logits = model.gcn(blocks, 'sto_predict', features)
                lv_loss = lv_loss_func(logits, labels) * 0.1
                optimizer.zero_grad()
                lv_loss.backward()
                optimizer.step()
                lv_history.append(lv_loss.item())
                torch.cuda.empty_cache()
            model.level_graph = model.level_graph.to(device)
            print('LV Training: Epoch {:04d} | Total Time {:.1f}s | Iter Loss {:.4f}'
                  .format(epoch, time() - time1, lv_loss.item()))

        # train kg & cf
        time1 = time()
        total_loss = 0

        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        n_batch = max(n_kg_batch, n_cf_batch)

        for iter in range(1, n_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item =\
                data.generate_cf_batch(data.train_user_dict)
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail =\
                data.generate_kg_batch(data.train_kg_dict)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)

                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            batch_losses = model.calc_loss(
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item,
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail)
            batch_loss = batch_losses[0].mean()
            history.append((batch_losses[0].mean().item(),
                            batch_losses[1].mean().item(),
                            batch_losses[2].mean().item()))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                print('KG & CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                      .format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        print('KG & CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'
              .format(epoch, n_batch, time() - time1, total_loss / n_batch))
        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            time1 = time()
            _, precision, recall, ndcg = evaluate(
                model, data.train_user_dict, data.test_user_dict,
                user_ids_batches, item_ids, data, args.K)
            print('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'
                  .format(epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            best_recall, should_stop = early_stopping(recall_list,
                                                      args.stopping_steps)

            if should_stop:
                break
            if recall_list.index(best_recall) == len(recall_list)-1:
                torch.save(model.state_dict(),
                           'ckpt/xue/kpcr_{}_best.pt'.format(args.kpcr_mode))
                best_epoch = epoch
    _, precision, recall, ndcg = evaluate(
        model, data.train_user_dict, data.test_user_dict,
        user_ids_batches, item_ids, data, args.K)
    print('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'
          .format(precision, recall, ndcg))
    print('Best Epoch : {}, Best Recall :{:.4f}'
          .format(best_epoch, best_recall))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)


if __name__ == '__main__':
    args = parse_args()
    main(args)
