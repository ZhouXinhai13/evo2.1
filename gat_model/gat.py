# coding=UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import layers  # 这个也需要改写
from src.gat_model.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=F.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = torch.cat(attns, dim=-1)
        
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = torch.cat(attns, dim=-1)
        
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = sum(out) / n_heads[-1]
        
        return logits


class HeteGAT_multi(nn.Module, BaseGAttN):
    def __init__(self, params):
        super(HeteGAT_multi, self).__init__()
        self.relation_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = 1
        self.hidden_size = 50
        self.batch_size = params['batch_size'] * params['num_rollouts']
        
        # 创建LSTM
        self.rnn_step = nn.LSTM(
            input_size=6 * self.hidden_size,
            hidden_size=6 * self.hidden_size,
            num_layers=self.LSTM_Layers,
            batch_first=True
        )
        
        # 用于inference1的dense层
        self.dense1 = nn.Linear(self.embedding_size * 3 + 1, 100)  # 假设time维度为1
        self.dense2 = nn.Linear(100, 1)
        
        # 用于inference2的dense层
        self.score_dense = nn.Linear(6 * self.hidden_size, self.entity_vocab_size)

    def inference1(self, input, entity_table, relation_table, time_in_x, bias_mat, mask_mat, nb_nodes, nb_input):
        """
        input: [nb_input, embedding_size]
        entity_table: [nb_nodes, embedding_size]
        relation_table: embedding table
        """
        # 关系嵌入查找
        rk = relation_table[bias_mat]  # [nb_input, nb_nodes, embedding_size]
        
        # 扩展input
        ei = input.unsqueeze(1)  # [nb_input, 1, embedding_size]
        ei = ei.expand(-1, nb_nodes, -1)  # [nb_input, nb_nodes, embedding_size]
        
        # 扩展entity
        ej = entity_table.unsqueeze(0)  # [1, nb_nodes, embedding_size]
        ej = ej.expand(nb_input, -1, -1)  # [nb_input, nb_nodes, embedding_size]
        
        # 扩展time
        t = time_in_x.unsqueeze(0).unsqueeze(0)  # [1, 1, time_dim]
        t = t.expand(nb_input, nb_nodes, -1)  # [nb_input, nb_nodes, time_dim]
        
        # 拼接特征
        cijk = torch.cat([ei, ej, rk, t], dim=2)  # [nb_input, nb_nodes, total_dim]
        
        # 两层dense
        mcijk = self.dense1(cijk)  # [nb_input, nb_nodes, 100]
        bijk = self.dense2(mcijk)  # [nb_input, nb_nodes, 1]
        bijk = bijk.squeeze(-1)  # [nb_input, nb_nodes]
        
        # 激活和attention
        bijk = F.leaky_relu(bijk)
        aijk = F.softmax(bijk + mask_mat * 1e9, dim=-1)  # [nb_input, nb_nodes]
        
        # 加权聚合
        eih = torch.bmm(aijk.unsqueeze(1), mcijk)  # [nb_input, 1, 100]
        
        return eih.squeeze(1) + input

    # def inference2(self, entity_table, relation_table, time_in_x, query, trace, 
    #                batch_size, nb_nodes, prev_state, range_h):
    #     """
    #     entity_table: [nb_entities, embedding_size]
    #     relation_table: [nb_relations, embedding_size]
    #     query: [batch_size]
    #     trace: [batch_size, path_length]
    #     prev_state: (h, c) tuple for LSTM
    #     """
    #     # Query embedding
    #     query_embedding = relation_table[query]  # [batch_size, embedding_size]
        
    #     # Time embedding
    #     t = time_in_x.unsqueeze(0)  # [1, time_dim]
    #     t = t.expand(batch_size, -1)  # [batch_size, time_dim]
        
    #     # Entity embeddings
    #     es = trace[:, 0]
    #     eo = trace[:, -1]
    #     es_embedding = entity_table[es]  # [batch_size, embedding_size]
        
    #     # RNN输入
    #     rnn_input = torch.cat([query_embedding, es_embedding, t], dim=1)  # [batch_size, total_dim]
    #     rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, total_dim]
        
    #     # LSTM forward
    #     output_f, state_f = self.rnn_step(rnn_input, prev_state)
    #     output_f = output_f.squeeze(1)  # [batch_size, hidden_size]
        
    #     # 计算分数
    #     score = self.score_dense(output_f)  # [batch_size, nb_nodes]
    #     score_softmax = F.softmax(score, dim=-1)
        
    #     # 提取目标实体分数
    #     indices = torch.stack([range_h, eo], dim=1)
    #     score_eo = score_softmax[indices[:, 0], indices[:, 1]]
        
    #     return score, state_f, score_eo
    # def inference2(self, entity_table, relation_table, time_in_x, query, trace, 
    #             batch_size, nb_nodes, prev_state, range_h):
    #     """
    #     entity_table: [nb_entities, embedding_size]
    #     relation_table: [nb_relations, embedding_size]
    #     query: [batch_size]
    #     trace: [batch_size, path_length]
    #     prev_state: (h, c) tuple for LSTM
    #     """
    #     # Query embedding
    #     query_embedding = relation_table[query]  # [batch_size, embedding_size]
        
    #     # Time embedding - ✅ 确保维度正确
    #     if time_in_x.dim() == 0:  # scalar
    #         t = time_in_x.unsqueeze(0).unsqueeze(0)  # [1, 1]
    #         t = t.expand(batch_size, -1)  # [batch_size, 1]
    #     elif time_in_x.dim() == 1:  # [batch_size]
    #         t = time_in_x.unsqueeze(1)  # [batch_size, 1]
    #     else:  # already [batch_size, time_dim]
    #         t = time_in_x
        
    #     # Entity embeddings
    #     es = trace[:, 0]
    #     eo = trace[:, -1]
    #     es_embedding = entity_table[es]  # [batch_size, embedding_size]
        
    #     # RNN输入 - ✅ 确保所有维度匹配
    #     rnn_input = torch.cat([query_embedding, es_embedding, t], dim=1)  # [batch_size, total_dim]
    #     rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, total_dim]
        
    #     # LSTM forward
    #     output_f, state_f = self.rnn_step(rnn_input, prev_state)
    #     output_f = output_f.squeeze(1)  # [batch_size, hidden_size]
        
    #     # 计算分数
    #     score = self.score_dense(output_f)  # [batch_size, nb_nodes]
    #     score_softmax = F.softmax(score, dim=-1)
        
    #     # ✅ 修复：提取目标实体分数的正确方式
    #     # 使用高级索引而不是构建 indices 矩阵
    #     score_eo = score_softmax[range_h, eo]  # [batch_size]
        
    #     return score, state_f, score_eo
    def inference2(self, entity_table, relation_table, time_in_x, query, trace, 
                batch_size, nb_nodes, prev_state, range_h):
        """
        entity_table: [nb_entities, 2*embedding_size]
        relation_table: [nb_relations, 2*embedding_size]
        query: [batch_size]
        trace: [batch_size, path_length]
        prev_state: (h, c) tuple for LSTM or None
        """
        # Query embedding
        query_embedding = relation_table[query]  # [batch_size, 2*embedding_size]
        
        # Time embedding - 确保维度正确
        if time_in_x.dim() == 0:  # scalar
            t = time_in_x.unsqueeze(0).unsqueeze(0)  # [1, 1]
            t = t.expand(batch_size, -1)  # [batch_size, 1]
        elif time_in_x.dim() == 1:  # [batch_size]
            t = time_in_x.unsqueeze(1)  # [batch_size, 1]
        else:  # already [batch_size, time_dim]
            t = time_in_x
        
        # Entity embeddings
        es = trace[:, 0]
        eo = trace[:, -1]
        es_embedding = entity_table[es]  # [batch_size, 2*embedding_size]
        eo_embedding = entity_table[eo]  # [batch_size, 2*embedding_size]
        
        # ✅ 修复：构建正确维度的 RNN 输入
        # LSTM 期望 6*hidden_size = 300 维度
        # query_embedding: 100, es_embedding: 100, eo_embedding: 100 = 300
        rnn_input = torch.cat([query_embedding, es_embedding, eo_embedding], dim=1)
        
        # 检查维度
        expected_dim = 6 * self.hidden_size  # 300
        if rnn_input.size(-1) != expected_dim:
            # 如果维度不匹配，用线性层调整
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(
                    rnn_input.size(-1), 
                    expected_dim
                ).to(rnn_input.device)
            rnn_input = self.input_projection(rnn_input)
        
        rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, expected_dim]
        
        # 初始化 LSTM state（如果为 None）
        if prev_state is None:
            h0 = torch.zeros(
                self.LSTM_Layers, batch_size, 6 * self.hidden_size,
                device=rnn_input.device
            )
            c0 = torch.zeros(
                self.LSTM_Layers, batch_size, 6 * self.hidden_size,
                device=rnn_input.device
            )
            prev_state = (h0, c0)
        
        # LSTM forward
        output_f, state_f = self.rnn_step(rnn_input, prev_state)
        output_f = output_f.squeeze(1)  # [batch_size, 6*hidden_size]
        
        # 计算分数
        score = self.score_dense(output_f)  # [batch_size, nb_nodes]
        score_softmax = F.softmax(score, dim=-1)
        
        # 提取目标实体分数
        score_eo = score_softmax[range_h, eo]  # [batch_size]
        
        return score, state_f, score_eo


class HeteGAT_no_coef(nn.Module, BaseGAttN):
    def __init__(self, nb_classes, hid_units, n_heads, mp_att_size=128):
        super(HeteGAT_no_coef, self).__init__()
        self.nb_classes = nb_classes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.mp_att_size = mp_att_size
        
        # 最后的分类层
        self.clf_layers = nn.ModuleList([
            nn.Linear(hid_units[-1] * n_heads[-2], nb_classes) 
            for _ in range(n_heads[-1])
        ])
    
    def forward(self, inputs, nb_nodes, training, attn_drop, ffd_drop,
                bias_mat_list, activation=F.elu, residual=False):
        embed_list = []
        
        for bias_mat in bias_mat_list:
            attns = []
            for _ in range(self.n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=self.hid_units[0], 
                                              activation=activation,
                                              in_drop=ffd_drop, 
                                              coef_drop=attn_drop, 
                                              residual=False))
            h_1 = torch.cat(attns, dim=-1)
            
            for i in range(1, len(self.hid_units)):
                attns = []
                for _ in range(self.n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=self.hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = torch.cat(attns, dim=-1)
            
            embed_list.append(h_1.squeeze().unsqueeze(1))
        
        # Metapath attention
        multi_embed = torch.cat(embed_list, dim=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, self.mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        
        # 分类层
        out = []
        for clf_layer in self.clf_layers:
            out.append(clf_layer(final_embed))
        logits = sum(out) / self.n_heads[-1]
        
        logits = logits.unsqueeze(0)
        return logits, final_embed, att_val


class HeteGAT(nn.Module, BaseGAttN):
    def __init__(self, nb_classes, hid_units, n_heads, mp_att_size=128):
        super(HeteGAT, self).__init__()
        self.nb_classes = nb_classes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.mp_att_size = mp_att_size
        
        # 最后的分类层
        self.clf_layers = nn.ModuleList([
            nn.Linear(hid_units[-1] * n_heads[-2], nb_classes) 
            for _ in range(n_heads[-1])
        ])
    
    def forward(self, inputs, nb_nodes, training, attn_drop, ffd_drop,
                bias_mat_list, activation=F.elu, residual=False, return_coef=False):
        embed_list = []
        coef_list = []
        
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            
            for _ in range(self.n_heads[0]):
                if return_coef:
                    a1, a2 = layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=self.hid_units[0], 
                                              activation=activation,
                                              in_drop=ffd_drop, 
                                              coef_drop=attn_drop, 
                                              residual=False,
                                              return_coef=return_coef)
                    attns.append(a1)
                    head_coef_list.append(a2)
                else:
                    attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=self.hid_units[0], 
                                                  activation=activation,
                                                  in_drop=ffd_drop, 
                                                  coef_drop=attn_drop, 
                                                  residual=False))
            
            if return_coef:
                head_coef = torch.cat(head_coef_list, dim=0)
                head_coef = head_coef.mean(dim=0)
                coef_list.append(head_coef)
            
            h_1 = torch.cat(attns, dim=-1)
            
            for i in range(1, len(self.hid_units)):
                attns = []
                for _ in range(self.n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=self.hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = torch.cat(attns, dim=-1)
            
            embed_list.append(h_1.squeeze().unsqueeze(1))
        
        # Metapath attention
        multi_embed = torch.cat(embed_list, dim=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, self.mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        
        # 分类层
        out = []
        for clf_layer in self.clf_layers:
            out.append(clf_layer(final_embed))
        logits = sum(out) / self.n_heads[-1]
        
        logits = logits.unsqueeze(0)
        
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val
