import numpy as np
import torch


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)),
                  axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def calc_metrics_at_k_(cf_scores, train_user_dict, test_user_dict,
                       user_ids, item_ids, num_users, K):
    """
    cf_scores: (n_eval_users, n_eval_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)],
                                    dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = np.array(train_user_dict[u])
        test_pos_item_list = np.array(test_user_dict[u])
        cf_scores[idx][train_pos_item_list-num_users] = 0
        test_pos_item_binary[idx][test_pos_item_list-num_users] = 1
    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    precision = precision_at_k_batch(binary_hit, K)
    recall = recall_at_k_batch(binary_hit, K)
    ndcg = ndcg_at_k_batch(binary_hit, K)
    return precision, recall, ndcg
