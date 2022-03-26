import collections
import os
import random
import re

import dgl
import numpy as np
import pandas as pd
import torch


class BaseLoader(object):

    def __init__(self, args):
        self.args = args
        if args.dataset == 'xue':
            self.data_path = './XuetangX/sampled/'
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.cf_train_data = pd.read_csv(
            os.path.join(self.data_path, 'cf_train.csv'), index_col=0)
        self.cf_test_data = pd.read_csv(
            os.path.join(self.data_path, 'cf_test.csv'), index_col=0)
        self.kg_data = pd.read_csv(
            os.path.join(self.data_path, 'kg_data.csv'), index_col=0)
        self.level_data = pd.read_csv(
            os.path.join(self.data_path, 'level_df.csv'), index_col=0)
        self.level_data.h = self.level_data.h.astype(int)
        self.level_data.t = self.level_data.t.astype(int)
        self.lec_level = np.load(
            os.path.join(self.data_path, 'lec_levels.npy'))

    def construct_cf_dict(self):
        self.train_user_dict = {}
        for rows in self.cf_train_data.iloc[:, [0, 2]].values:
            if rows[0] in self.train_user_dict.keys():
                self.train_user_dict[rows[0]].append(rows[1])
            else:
                self.train_user_dict[rows[0]] = [rows[1]]
        self.test_user_dict = {}
        for rows in self.cf_test_data.iloc[:, [0, 2]].values:
            if rows[0] in self.test_user_dict.keys():
                self.test_user_dict[rows[0]].append(rows[1])
            else:
                self.test_user_dict[rows[0]] = [rows[1]]

    def construct_kg_dict(self):
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        self.test_kg_dict = collections.defaultdict(list)
        self.test_relation_dict = collections.defaultdict(list)
        for row in self.kg_test_data.iterrows():
            h, r, t = row[1]
            self.test_kg_dict[h].append((t, r))
            self.test_relation_dict[r].append((h, t))

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break
            pos_item_idx = np.random.randint(
                low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            neg_item_id = np.random.randint(
                low=0+self.n_users, high=self.n_items+self.n_users, size=1)[0]
            if neg_item_id not in pos_items and \
               neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict):
        exist_users = user_dict.keys()
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users)
                          for _ in range(self.cf_batch_size)]
        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break
            pos_triple_idx = np.random.randint(
                low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and\
               tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head,
                                 relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(
                low=self.n_users, high=self.n_entities+self.n_users, size=1)[0]
            if (tail, relation) not in pos_triples and\
               tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(self.kg_batch_size)]
        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(
                kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)

        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


class DataLoader_LS(BaseLoader):
    def __init__(self,args):
        super().__init__(args)

        self.n_users = max(max(self.cf_train_data.h),
                           max(self.cf_test_data.h)) + 1
        self.n_items = max(max(self.cf_train_data.t),
                           max(self.cf_test_data.t)) + 1 - self.n_users
        self.n_cf_train = len(self.cf_train_data)
        self.n_cf_test = len(self.cf_test_data)
        self.n_relations = max(self.kg_data['r'])+1
        self.n_users_entities = max(max(self.kg_data['h']),
                                    max(self.kg_data['t'])) + 1

        self.kg_train_data = pd.concat(
            [self.kg_data, self.cf_train_data], ignore_index=True)
        self.kg_test_data = pd.concat(
            [self.kg_data, self.cf_test_data], ignore_index=True)        
        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)

        self.construct_cf_dict()
        self.construct_kg_dict()
        self.level_graph = self.create_level_graph(self.level_data)

    def create_level_graph(self, df):
        g = dgl.graph((torch.LongTensor(df.h.values),
                       torch.LongTensor(df.t.values)))
        if self.args.dataset == 'ebs' or self.args.dataset == 'usecase':
            g.ndata['level'] = torch.LongTensor(
                df.iloc[:, [0, 1]].sort_values('h').drop_duplicates().h_level.values.tolist() +
                (self.lec_level).tolist())
            g.ndata['feat'] = torch.eye(self.n_users+self.n_items)
        elif self.args.dataset == 'xue':
            lvs = df.iloc[:, [0, 1]].sort_values('h').\
                    drop_duplicates(subset=['h']).h_level.values.tolist()
            g.ndata['level'] = torch.LongTensor(
                [eval(re.sub(r'\n', '', re.sub(r'\.',r',',_))) for _ in lvs] +
                self.lec_level.tolist())

            g.ndata['feat'] = torch.eye(self.n_users+self.n_items)
        return g

    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(self.kg_batch_size)]
        batch_relation, batch_pos_tail = [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)

        return batch_head, batch_relation, batch_pos_tail


class DataLoader_S(BaseLoader):
    def __init__(self,args):
        super().__init__(args)

        self.n_users = max(max(self.cf_train_data.h),
                           max(self.cf_test_data.h)) + 1
        self.n_items = max(max(self.cf_train_data.t),
                           max(self.cf_test_data.t)) + 1 - self.n_users
        self.n_cf_train = len(self.cf_train_data)
        self.n_cf_test = len(self.cf_test_data)
        self.n_relations = max(self.kg_data['r'])+1
        self.n_users_entities = max(max(self.kg_data['h']),
                                    max(self.kg_data['t'])) + 1

        self.kg_train_data = pd.concat(
            [self.kg_data, self.cf_train_data], ignore_index=True)
        self.kg_test_data = pd.concat(
            [self.kg_data, self.cf_test_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)

        self.construct_cf_dict()
        self.construct_kg_dict()

    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(self.kg_batch_size)]
        batch_relation, batch_pos_tail = [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)

        return batch_head, batch_relation, batch_pos_tail


class DataLoader_Si(BaseLoader):
    def __init__(self,args):
        super().__init__(args)
        self.kg_data = self.kg_data[self.kg_data.r <= 2]

        self.n_users = max(max(self.cf_train_data.h),
                           max(self.cf_test_data.h)) + 1
        self.n_items = max(max(self.cf_train_data.t),
                           max(self.cf_test_data.t)) + 1 - self.n_users
        self.n_cf_train = len(self.cf_train_data)
        self.n_cf_test = len(self.cf_test_data)
        self.n_relations = max(self.kg_data['r'])+1
        self.n_users_entities = max(max(self.kg_data['h']),
                                    max(self.kg_data['t'])) + 1

        self.kg_train_data = pd.concat(
            [self.kg_data, self.cf_train_data], ignore_index=True)
        self.kg_test_data = pd.concat(
            [self.kg_data, self.cf_test_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)

        self.construct_cf_dict()
        self.construct_kg_dict()

    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(self.kg_batch_size)]
        batch_relation, batch_pos_tail = [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)

        return batch_head, batch_relation, batch_pos_tail


class ItemDataLoader(BaseLoader):
    def __init__(self, args):
        super().__init__(args)

        self.n_users = max(max(self.cf_train_data.h),
                           max(self.cf_test_data.h)) + 1
        self.n_items = max(max(self.cf_train_data.t),
                           max(self.cf_test_data.t)) + 1 - self.n_users
        self.n_cf_train = len(self.cf_train_data)
        self.n_cf_test = len(self.cf_test_data)

        self.kg_data = self.kg_data[self.kg_data.h >= self.n_users]
        self.n_kg_data = len(self.kg_data)
        self.n_relations = len(self.kg_data.r.unique())
        self.n_entities = len(list(set(self.kg_data.h).union(
                                                        set(self.kg_data.t))))
        # rearange  id
        self.rel_remapping = {}
        for ri, i in enumerate(self.kg_data.r.unique()):
            self.rel_remapping[i] = ri
        self.ent_remapping = {}
        for ei, i in enumerate(sorted(list(set(self.kg_data.h).union(
                                                    set(self.kg_data.t))))):
            self.ent_remapping[i] = ei + self.n_users
        self.kg_data.h = self.kg_data.h.map(self.ent_remapping)
        self.kg_data.r = self.kg_data.r.map(self.rel_remapping)
        self.kg_data.t = self.kg_data.t.map(self.ent_remapping)

        self.construct_cf_dict()
        # construct kg dict
        self.kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)
        for row in self.kg_data.iterrows():
            h, r, t = row[1]
            self.kg_dict[h].append((t, r))
            self.relation_dict[r].append((h, t))

    def generate_kg_batch_(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(self.kg_batch_size)]
        batch_relation, batch_pos_tail = [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)

        return batch_head, batch_relation, batch_pos_tail
