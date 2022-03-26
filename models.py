from dgl.nn import GraphConv
import torch
from torch import nn
from torch.nn import functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, blocks, mode, x):
        if mode == 'sto_predict':
            x = self.conv1(blocks[0], x)
            x = F.relu(x)
            x = self.conv2(blocks[1], x)
            return x
        elif mode == 'predict':
            x = self.conv1(blocks, x)
            x = F.relu(x)
            x = self.conv2(blocks, x)
            return x
        elif mode == 'embedding':
            x = self.conv1(blocks, x)
            return x


class KPCR_LS(nn.Module):
    def __init__(self, args, g,
                 n_users, n_items, n_entities, n_relations):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.embed_dim

        self.embed_shape = args.embed_shape
        self.kernel_size = args.kernel_size
        self.out_channels = args.out_channels
        self.emb_drop = eval(args.dropout)[0]
        self.feat_drop = eval(args.dropout)[1]
        self.h_drop = eval(args.dropout)[2]
        self.label_smoothing = args.label_smoothing
        self.emb_dim1 = self.embed_shape
        self.emb_dim2 = self.embed_dim // self.emb_dim1

        self.cf_l2_loss_lambda = args.cf_l2_loss_lambda
        self.kg_l2_loss_lambda = args.kg_l2_loss_lambda

        # Node Classification part
        if args.dataset == 'xue':
            self.num_lv = 30
        elif args.dataset == 'ebs' or args.dataset == 'usecase':
            self.num_lv = 6
        self.level_graph = g
        self.gcn = GCN(self.n_users + self.n_items,
                       args.embed_dim, self.num_lv)
        # ConvE part
        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        self.emb_drop_layer = nn.Dropout(self.emb_drop)
        self.feat_drop_layer = nn.Dropout(self.feat_drop)
        self.h_drop_layer = nn.Dropout(self.h_drop)

        self.conv1 = nn.Conv2d(1, self.out_channels,
                               (self.kernel_size, self.kernel_size), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)
        self.register_parameter('b',
                                nn.Parameter(torch.zeros(self.n_entities)))
        self.fc = nn.Linear(
            (2*self.emb_dim1 - self.kernel_size + 1) *
            (self.emb_dim2 - self.kernel_size + 1) *
            self.out_channels, self.embed_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embed.weight,
                                gain=nn.init.calculate_gain('relu'))

    def calc_kg_loss(self, h, r, pos_t):
        h_embed = self.entity_embed(h).\
                  view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_embed = self.relation_embed(r).\
            view(-1, 1, self.emb_dim1, self.emb_dim2)

        x = torch.cat([h_embed, r_embed],2)
        x = self.bn0(x)
        x = self.emb_drop_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.h_drop_layer(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = torch.mm(x, self.entity_embed.weight.transpose(1,0))
        x += self.b.expand_as(x)

        bce = nn.BCEWithLogitsLoss()
        label = torch.eye(self.n_entities)[pos_t].to(pos_t.device)
        label = label * (1.0 - self.label_smoothing) + (1.0/self.n_entities)

        kg_loss = bce(x, label)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + \
            _L2_loss_mean(self.conv1.weight) + _L2_loss_mean(self.fc.weight)
        loss = kg_loss + self.kg_l2_loss_lambda * l2_loss

        return loss

    # CF part
    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        level_embed = self.gcn(
            self.level_graph, 'embedding',
            torch.eye(self.n_users+self.n_items,
                      device=self.level_graph.device)).to(user_ids.device)
        user_embed = self.entity_embed(user_ids) + level_embed[user_ids]
        item_pos_embed = self.entity_embed(item_pos_ids) +\
            level_embed[item_pos_ids]
        item_neg_embed = self.entity_embed(item_neg_ids) +\
            level_embed[item_neg_ids]
        user_embed = self.entity_embed(user_ids) + level_embed[user_ids]
        item_pos_embed = self.entity_embed(item_pos_ids) +\
            level_embed[item_pos_ids]
        item_neg_embed = self.entity_embed(item_neg_ids) +\
            level_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2_loss_lambda * l2_loss

        return loss

    def predict(self, user_ids, item_ids):
        level_embed = self.gcn(
            self.level_graph, 'embedding',
            torch.eye(self.n_users+self.n_items,
                      device=self.level_graph.device)).to(user_ids.device)
        user_embed = self.entity_embed(user_ids) + level_embed[user_ids]
        item_embed = self.entity_embed(item_ids) + level_embed[item_ids]
        user_embed = self.entity_embed(user_ids) + level_embed[user_ids]
        item_embed = self.entity_embed(item_ids) + level_embed[item_ids]
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return cf_score

    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids, h, r, pos_t):
        kg_loss = self.calc_kg_loss(h, r, pos_t)
        cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
        loss = 0.3*kg_loss + 0.7*cf_loss

        return loss, kg_loss, cf_loss


class KPCR_S(nn.Module):
    def __init__(self, args,
                 n_users, n_items, n_entities, n_relations):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.embed_dim

        self.embed_shape = args.embed_shape
        self.kernel_size = args.kernel_size
        self.out_channels = args.out_channels
        self.emb_drop = eval(args.dropout)[0]
        self.feat_drop = eval(args.dropout)[1]
        self.h_drop = eval(args.dropout)[2]
        self.label_smoothing = args.label_smoothing
        self.emb_dim1 = self.embed_shape
        self.emb_dim2 = self.embed_dim // self.emb_dim1

        self.cf_l2_loss_lambda = args.cf_l2_loss_lambda
        self.kg_l2_loss_lambda = args.kg_l2_loss_lambda

        # ConvE part
        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        self.emb_drop_layer = nn.Dropout(self.emb_drop)
        self.feat_drop_layer = nn.Dropout(self.feat_drop)
        self.h_drop_layer = nn.Dropout(self.h_drop)

        self.conv1 = nn.Conv2d(1, self.out_channels,
                               (self.kernel_size, self.kernel_size), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)
        self.register_parameter('b',
                                nn.Parameter(torch.zeros(self.n_entities)))
        self.fc = nn.Linear((2*self.emb_dim1 - self.kernel_size + 1) *
                            (self.emb_dim2 - self.kernel_size + 1) *
                            self.out_channels, self.embed_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embed.weight,
                                gain=nn.init.calculate_gain('relu'))

    def calc_kg_loss(self, h, r, pos_t):
        h_embed = self.entity_embed(h).\
            view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_embed = self.relation_embed(r).\
            view(-1, 1, self.emb_dim1, self.emb_dim2)

        x = torch.cat([h_embed, r_embed], 2)
        x = self.bn0(x)
        x = self.emb_drop_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.h_drop_layer(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = torch.mm(x, self.entity_embed.weight.transpose(1, 0))
        x += self.b.expand_as(x)

        bce = nn.BCEWithLogitsLoss()
        label = torch.eye(self.n_entities)[pos_t].to(pos_t.device)
        label = label * (1.0 - self.label_smoothing) + (1.0/self.n_entities)

        kg_loss = bce(x, label)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + \
            _L2_loss_mean(self.conv1.weight) + _L2_loss_mean(self.fc.weight)
        loss = kg_loss + self.kg_l2_loss_lambda * l2_loss

        return loss

    # CF part
    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        user_embed = self.entity_embed(user_ids)
        item_pos_embed = self.entity_embed(item_pos_ids)
        item_neg_embed = self.entity_embed(item_neg_ids)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) +\
            _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2_loss_lambda * l2_loss

        return loss

    def predict(self, user_ids, item_ids):
        user_embed = self.entity_embed(user_ids)
        item_embed = self.entity_embed(item_ids)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return cf_score

    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids, h, r, pos_t):
        kg_loss = self.calc_kg_loss(h,r,pos_t)
        cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
        loss = 0.4*kg_loss + 0.6*cf_loss

        return loss, kg_loss, cf_loss
