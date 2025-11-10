# # # coding=UTF-8
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F


# # class Agent(nn.Module):
# #     def __init__(self, params):
# #         super(Agent, self).__init__()
        
# #         self.action_vocab_size = len(params['relation_vocab'])
# #         self.entity_vocab_size = len(params['entity_vocab'])
# #         self.time_vocab_size = len(params['time_vocab'])
# #         self.embedding_size = params['embedding_size']
# #         self.hidden_size = params['hidden_size']
# #         self.ePAD = params['entity_vocab']['PAD']
# #         self.rPAD = params['relation_vocab']['PAD']
        
# #         self.train_entities = params['train_entity_embeddings']
# #         self.train_relations = params['train_relation_embeddings']
        
# #         self.num_rollouts = params['num_rollouts']
# #         self.test_rollouts = params['test_rollouts']
# #         self.LSTM_Layers = params['LSTM_layers']
# #         self.batch_size = params['batch_size'] * params['num_rollouts']
        
# #         self.entity_embedding_size = self.embedding_size
# #         self.use_entity_embeddings = params['use_entity_embeddings']
        
# #         if self.use_entity_embeddings:
# #             self.m = 4 + 2
# #         else:
# #             self.m = 2 + 2
        
# #         # Embedding tables
# #         if params['use_entity_embeddings']:
# #             self.entity_lookup_table = nn.Embedding(
# #                 self.entity_vocab_size, 
# #                 2 * self.entity_embedding_size
# #             )
# #             nn.init.xavier_normal_(self.entity_lookup_table.weight)
# #         else:
# #             self.entity_lookup_table = nn.Embedding(
# #                 self.entity_vocab_size, 
# #                 2 * self.entity_embedding_size
# #             )
# #             nn.init.zeros_(self.entity_lookup_table.weight)
        
# #         self.entity_lookup_table.weight.requires_grad = self.train_entities
        
# #         self.relation_lookup_table = nn.Embedding(
# #             self.action_vocab_size, 
# #             2 * self.embedding_size
# #         )
# #         nn.init.xavier_normal_(self.relation_lookup_table.weight)
# #         self.relation_lookup_table.weight.requires_grad = self.train_relations
        
# #         self.time_lookup_table = nn.Embedding(
# #             self.time_vocab_size, 
# #             2 * self.entity_embedding_size
# #         )
# #         nn.init.xavier_normal_(self.time_lookup_table.weight)
# #         self.time_lookup_table.weight.requires_grad = self.train_entities
        
# #         # LSTM
# #         self.policy_step = nn.LSTM(
# #             input_size=self.m * self.embedding_size,
# #             hidden_size=self.m * self.hidden_size,
# #             num_layers=self.LSTM_Layers,
# #             batch_first=True
# #         )
        
# #         # MLP for policy
# #         if self.use_entity_embeddings:
# #             policy_input_dim = (self.m * self.hidden_size +  # state
# #                             2 * self.entity_embedding_size +  # prev_entity
# #                             2 * self.embedding_size +  # query
# #                             2 * self.entity_embedding_size)  # time
# #         else:
# #             policy_input_dim = (self.m * self.hidden_size +  # state (no prev_entity)
# #                             2 * self.embedding_size +  # query
# #                             2 * self.entity_embedding_size)  # time

# #         self.policy_mlp = nn.Linear(
# #             policy_input_dim,
# #             self.m * self.embedding_size
# #         )
        
# #         # Dummy start relation
# #         self.dummy_start_label = params['relation_vocab']['DUMMY_START_RELATION']
    
# #     def get_mem_shape(self):
# #         return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)
    
# #     def initialize_embeddings(self, entity_emb=None, relation_emb=None, time_emb=None):
# #         """Initialize embeddings from pretrained weights"""
# #         if entity_emb is not None:
# #             self.entity_lookup_table.weight.data.copy_(torch.from_numpy(entity_emb))
# #         if relation_emb is not None:
# #             self.relation_lookup_table.weight.data.copy_(torch.from_numpy(relation_emb))
# #         if time_emb is not None:
# #             self.time_lookup_table.weight.data.copy_(torch.from_numpy(time_emb))
    
# #     def action_encoder(self, next_relations, next_entities, next_times):
# #         """Encode actions (relation, entity, time) into embeddings"""
# #         relation_embedding = self.relation_lookup_table(next_relations)
# #         entity_embedding = self.entity_lookup_table(next_entities)
# #         time_embedding = self.time_lookup_table(next_times)
# #         # ❌ 消融：时间嵌入全零
# #         # time_embedding = torch.zeros_like(
# #         #     self.time_lookup_table(next_times)
# #         # )        
# #         if self.use_entity_embeddings:
# #             action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
# #         else:
# #             action_embedding = relation_embedding
# #         action_embedding = torch.cat([action_embedding, time_embedding], dim=-1)
        
# #         return action_embedding
    
# #     def step(self, next_relations, next_entities, next_times, prev_state, 
# #             prev_relation, prev_time, query_embedding, query_time,
# #             current_entities, label_action, range_arr, first_step_of_test):
# #         """
# #         Single step of the agent
        
# #         Args:
# #             next_relations: [batch_size, max_num_actions]
# #             next_entities: [batch_size, max_num_actions]
# #             next_times: [batch_size, max_num_actions]
# #             prev_state: tuple of (h, c) from LSTM
# #             prev_relation: [batch_size]
# #             prev_time: [batch_size]
# #             query_embedding: [batch_size, embedding_size]
# #             query_time: [batch_size]
# #             current_entities: [batch_size]
# #             label_action: [batch_size] or None (for supervised training)
# #             range_arr: [batch_size]
# #             first_step_of_test: bool
# #         """
# #         # Encode previous action
# #         prev_action_embedding = self.action_encoder(
# #             prev_relation.unsqueeze(1),
# #             current_entities.unsqueeze(1),
# #             prev_time.unsqueeze(1)
# #         ).squeeze(1)
        
# #         # LSTM step
# #         prev_action_embedding = prev_action_embedding.unsqueeze(1)
# #         output, new_state = self.policy_step(prev_action_embedding, prev_state)
# #         output = output.squeeze(1)
        
# #         # Get current entity embedding
# #         prev_entity = self.entity_lookup_table(current_entities)
        
# #         # Concatenate state
# #         if self.use_entity_embeddings:
# #             state = torch.cat([output, prev_entity], dim=-1)
# #         else:
# #             state = output
        
# #         # Encode candidate actions
# #         candidate_action_embeddings = self.action_encoder(
# #             next_relations, next_entities, next_times
# #         )
        
# #         # Concatenate with query
# #         query_time_emb = self.time_lookup_table(query_time)
# #         # query_time_emb = torch.zeros_like(self.time_lookup_table(query_time))
# #         # ✅ 调试输出
# #         print(f"Query time IDs: {query_time[:5]}")
# #         print(f"Query time emb mean: {query_time_emb.mean().item():.4f}")
# #         print(f"Query time emb std: {query_time_emb.std().item():.4f}")
# #         state_query_concat = torch.cat([state, query_embedding], dim=-1)
# #         state_query_time_concat = torch.cat([state_query_concat, query_time_emb], dim=-1)
        
# #         # Policy MLP
# #         output = F.relu(self.policy_mlp(state_query_time_concat))
# #         output_expanded = output.unsqueeze(1)
        
# #         # Calculate scores
# #         prelim_scores = torch.sum(
# #             candidate_action_embeddings * output_expanded, 
# #             dim=2
# #         )
        
# #         # Mask padding
# #         mask = (next_relations == self.rPAD)
# #         scores = prelim_scores.masked_fill(mask, -1e9)
        
# #         # Sample action
# #         action_dist = torch.distributions.Categorical(logits=scores)
# #         action = action_dist.sample()
        
# #         # Calculate loss
# #         # 修复: 处理 label_action 和 scores 的 batch size 不匹配问题
# #         if label_action is None or label_action.size(0) != scores.size(0):
# #             # RL training mode: use negative log probability of sampled action
# #             # This will be weighted by advantage later in REINFORCE
# #             loss = -action_dist.log_prob(action)
# #         else:
# #             # Supervised training mode: use cross entropy with provided labels
# #             loss = F.cross_entropy(scores, label_action, reduction='none')
        
# #         # Get chosen relation and time
# #         batch_indices = torch.arange(next_relations.size(0), device=next_relations.device)
# #         chosen_relation = next_relations[batch_indices, action]
# #         chosen_time = next_times[batch_indices, action]
        
# #         return loss, new_state, F.log_softmax(scores, dim=-1), action, chosen_relation, chosen_time

# #     def forward(self, candidate_relation_sequence, candidate_entity_sequence, 
# #                 candidate_time_sequence, current_entities, path_label, 
# #                 query_relation, query_time, range_arr, first_step_of_test, T=3):
# #         """
# #         Unroll the agent for T steps
        
# #         Returns:
# #             all_loss: list of losses per step
# #             all_logits: list of log probabilities per step
# #             action_idx: list of actions taken per step
# #         """
# #         batch_size = query_relation.size(0)
# #         device = query_relation.device
        
# #         # Initialize
# #         query_embedding = self.relation_lookup_table(query_relation)
        
# #         # Initial LSTM state
# #         h0 = torch.zeros(
# #             self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
# #             device=device
# #         )
# #         c0 = torch.zeros(
# #             self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
# #             device=device
# #         )
# #         state = (h0, c0)
        
# #         # Dummy start
# #         prev_relation = torch.ones(batch_size, dtype=torch.long, device=device) * self.dummy_start_label
# #         prev_time = torch.zeros(batch_size, dtype=torch.long, device=device)
        
# #         all_loss = []
# #         all_logits = []
# #         action_idx = []
        
# #         for t in range(T):
# #             next_possible_relations = candidate_relation_sequence[t]
# #             next_possible_entities = candidate_entity_sequence[t]
# #             next_possible_times = candidate_time_sequence[t]
# #             current_entities_t = current_entities[t]
# #             path_label_t = path_label[t]
            
# #             loss, state, logits, idx, chosen_relation, chosen_time = self.step(
# #                 next_possible_relations,
# #                 next_possible_entities,
# #                 next_possible_times,
# #                 state,
# #                 prev_relation,
# #                 prev_time,
# #                 query_embedding,
# #                 query_time,
# #                 current_entities_t,
# #                 label_action=path_label_t,
# #                 range_arr=range_arr,
# #                 first_step_of_test=first_step_of_test
# #             )
            
# #             all_loss.append(loss)
# #             all_logits.append(logits)
# #             action_idx.append(idx)
            
# #             prev_relation = chosen_relation
# #             prev_time = chosen_time
        
# #         return all_loss, all_logits, action_idx
# # coding=UTF-8
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Agent(nn.Module):
#     def __init__(self, params):
#         super(Agent, self).__init__()
        
#         self.action_vocab_size = len(params['relation_vocab'])
#         self.entity_vocab_size = len(params['entity_vocab'])
#         self.time_vocab_size = len(params['time_vocab'])
#         self.embedding_size = params['embedding_size']
#         self.hidden_size = params['hidden_size']
#         self.ePAD = params['entity_vocab']['PAD']
#         self.rPAD = params['relation_vocab']['PAD']
        
#         self.train_entities = params['train_entity_embeddings']
#         self.train_relations = params['train_relation_embeddings']
        
#         self.num_rollouts = params['num_rollouts']
#         self.test_rollouts = params['test_rollouts']
#         self.LSTM_Layers = params['LSTM_layers']
#         self.batch_size = params['batch_size'] * params['num_rollouts']
        
#         self.entity_embedding_size = self.embedding_size
#         self.use_entity_embeddings = params['use_entity_embeddings']
        
#         if self.use_entity_embeddings:
#             self.m = 4 + 2
#         else:
#             self.m = 2 + 2
        
#         # ===== Embedding tables =====
#         if params['use_entity_embeddings']:
#             self.entity_lookup_table = nn.Embedding(
#                 self.entity_vocab_size, 
#                 2 * self.entity_embedding_size
#             )
#             nn.init.xavier_normal_(self.entity_lookup_table.weight)
#         else:
#             self.entity_lookup_table = nn.Embedding(
#                 self.entity_vocab_size, 
#                 2 * self.entity_embedding_size
#             )
#             nn.init.zeros_(self.entity_lookup_table.weight)
        
#         self.entity_lookup_table.weight.requires_grad = self.train_entities
        
#         self.relation_lookup_table = nn.Embedding(
#             self.action_vocab_size, 
#             2 * self.embedding_size
#         )
#         nn.init.xavier_normal_(self.relation_lookup_table.weight)
#         self.relation_lookup_table.weight.requires_grad = self.train_relations
        
#         # ===== 🔥 核心改动：相对时间编码器 =====
#         # 替代原来的 time_lookup_table
#         self.time_diff_encoder = nn.Sequential(
#             # 第一层：将标量时间差映射到高维空间
#             nn.Linear(1, 2 * self.entity_embedding_size),
#             # ReLU激活：
#             # 1. 引入非线性，学习复杂的时间模式
#             # 2. 稀疏激活，只关注重要特征
#             # 3. 对"未来"(负数)有天然的抑制作用
#             nn.ReLU(),
#             # 第二层：进一步变换，提取高阶时间特征
#             nn.Linear(2 * self.entity_embedding_size, 2 * self.entity_embedding_size)
#         )
        
#         # 初始化MLP权重（使用Xavier，保证梯度流）
#         for layer in self.time_diff_encoder:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_normal_(layer.weight)
#                 nn.init.zeros_(layer.bias)
        
#         # ===== LSTM =====
#         self.policy_step = nn.LSTM(
#             input_size=self.m * self.embedding_size,
#             hidden_size=self.m * self.hidden_size,
#             num_layers=self.LSTM_Layers,
#             batch_first=True
#         )
        
#         # ===== MLP for policy =====
#         if self.use_entity_embeddings:
#             policy_input_dim = (self.m * self.hidden_size +  # state
#                             2 * self.entity_embedding_size +  # prev_entity
#                             2 * self.embedding_size +  # query
#                             2 * self.entity_embedding_size)  # time
#         else:
#             policy_input_dim = (self.m * self.hidden_size +  # state (no prev_entity)
#                             2 * self.embedding_size +  # query
#                             2 * self.entity_embedding_size)  # time

#         self.policy_mlp = nn.Linear(
#             policy_input_dim,
#             self.m * self.embedding_size
#         )
        
#         # Dummy start relation
#         self.dummy_start_label = params['relation_vocab']['DUMMY_START_RELATION']
    
#     def get_mem_shape(self):
#         return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)
    
#     def initialize_embeddings(self, entity_emb=None, relation_emb=None, time_emb=None):
#         """Initialize embeddings from pretrained weights"""
#         if entity_emb is not None:
#             self.entity_lookup_table.weight.data.copy_(torch.from_numpy(entity_emb))
#         if relation_emb is not None:
#             self.relation_lookup_table.weight.data.copy_(torch.from_numpy(relation_emb))
#         # time_emb参数保留但不使用（已改用MLP）
    
#     def encode_action_with_relative_time(self, relations, entities, times, query_time):
#         """
#         统一的动作编码函数：使用相对时间
        
#         Args:
#             relations: [batch_size, num_actions] 或 [batch_size, 1]
#             entities: [batch_size, num_actions] 或 [batch_size, 1]
#             times: [batch_size, num_actions] 或 [batch_size, 1]
#             query_time: [batch_size]
        
#         Returns:
#             action_embeddings: [batch_size, num_actions, emb_dim]
        
#         设计说明：
#         1. 计算相对时间差而非使用绝对时间戳
#         2. 归一化到[-1, 1]范围，提高训练稳定性
#         3. 通过MLP编码，学习非线性时间模式
#         """
#         # 1. 编码关系和实体（保持不变）
#         relation_emb = self.relation_lookup_table(relations)
#         entity_emb = self.entity_lookup_table(entities)
        
#         # 2. 🔥 计算相对时间差
#         # query_time: [batch_size]
#         # times: [batch_size, num_actions]
#         # 语义：time_diff > 0 表示候选动作在过去（可选）
#         #      time_diff < 0 表示候选动作在未来（不可选）
#         query_time_expanded = query_time.unsqueeze(1).float()  # [B, 1]
#         time_diffs = query_time_expanded - times.float()  # [B, num_actions]
        
#         # 3. 🔥 归一化时间差
#         # 原因：
#         # - 数值稳定性：防止MLP输入过大
#         # - 特征对齐：与实体/关系嵌入的值域匹配
#         # - 泛化能力：使模型能处理训练集外的时间跨度
#         time_diffs_normalized = time_diffs / self.time_vocab_size  # 缩放到约[-1, 1]
        
#         # 4. 🔥 通过MLP编码相对时间
#         # 输入：[B, num_actions] → 扩展为 [B, num_actions, 1]
#         # 输出：[B, num_actions, 2*entity_embedding_size]
#         time_diffs_expanded = time_diffs_normalized.unsqueeze(-1)
#         time_emb = self.time_diff_encoder(time_diffs_expanded)
        
#         # 5. 拼接所有特征
#         if self.use_entity_embeddings:
#             action_emb = torch.cat([relation_emb, entity_emb], dim=-1)
#         else:
#             action_emb = relation_emb
        
#         action_emb = torch.cat([action_emb, time_emb], dim=-1)
        
#         return action_emb
    
#     def step(self, next_relations, next_entities, next_times, prev_state, 
#             prev_relation, prev_time, query_embedding, query_time,
#             current_entities, label_action, range_arr, first_step_of_test):
#         """
#         Single step of the agent
        
#         核心改进：
#         1. prev_action 和 candidate_action 都使用相对时间编码
#         2. 保证LSTM输入和Policy输入的时间表示一致
#         3. 添加未来动作的软惩罚（训练时）
#         """
        
#         # ===== 🔥 修复1：prev_action 也用相对时间 =====
#         # 这确保LSTM看到的时间信息与policy一致
#         prev_action_embedding = self.encode_action_with_relative_time(
#             prev_relation.unsqueeze(1),      # [B, 1]
#             current_entities.unsqueeze(1),   # [B, 1]
#             prev_time.unsqueeze(1),          # [B, 1]
#             query_time                       # [B]
#         ).squeeze(1)  # [B, emb_dim]
        
#         # LSTM step
#         prev_action_embedding = prev_action_embedding.unsqueeze(1)
#         output, new_state = self.policy_step(prev_action_embedding, prev_state)
#         output = output.squeeze(1)
        
#         # Get current entity embedding
#         prev_entity = self.entity_lookup_table(current_entities)
        
#         # Concatenate state
#         if self.use_entity_embeddings:
#             state = torch.cat([output, prev_entity], dim=-1)
#         else:
#             state = output
        
#         # ===== 🔥 修复2：候选动作也用相对时间 =====
#         candidate_action_embeddings = self.encode_action_with_relative_time(
#             next_relations,  # [B, max_actions]
#             next_entities,   # [B, max_actions]
#             next_times,      # [B, max_actions]
#             query_time       # [B]
#         )  # [B, max_actions, emb_dim]
        
#         # ===== 🔥 修复3：查询时间也用相对表示（相对于自己=0）=====
#         # 创建全零的时间差（表示"当前时刻"）
#         batch_size = query_time.size(0)
#         zero_diffs = torch.zeros(batch_size, 1, 1, device=query_time.device)
#         query_time_emb = self.time_diff_encoder(zero_diffs).squeeze(1)  # [B, emb_dim]
        
#         # Concatenate with query
#         state_query_concat = torch.cat([state, query_embedding], dim=-1)
#         state_query_time_concat = torch.cat([state_query_concat, query_time_emb], dim=-1)
        
#         # Policy MLP
#         output = F.relu(self.policy_mlp(state_query_time_concat))
#         output_expanded = output.unsqueeze(1)
        
#         # Calculate scores
#         prelim_scores = torch.sum(
#             candidate_action_embeddings * output_expanded, 
#             dim=2
#         )
        
#         # ===== 🔥 新增：软惩罚未来动作 =====
#         # 这是一个可学习的信号，不是硬过滤
#         if self.training:
#             query_time_expanded = query_time.unsqueeze(1).float()
#             time_diffs = query_time_expanded - next_times.float()
            
#             # 未来动作（time_diff < 0）施加负分
#             # 惩罚值-1.0：足够强但不会完全压制（让模型学习）
#             future_penalty = -1.0 * (time_diffs < 0).float()
#             prelim_scores = prelim_scores + future_penalty
        
#         # Mask padding
#         mask = (next_relations == self.rPAD)
#         scores = prelim_scores.masked_fill(mask, -1e9)
        
#         # Sample action
#         action_dist = torch.distributions.Categorical(logits=scores)
#         action = action_dist.sample()
        
#         # # Calculate loss
#         # if label_action is None or label_action.size(0) != scores.size(0):
#         #     # RL training mode
#         #     loss = -action_dist.log_prob(action)
#         # else:
#         #     # Supervised training mode
#         #     loss = F.cross_entropy(scores, label_action, reduction='none')
#         # ===== 🔥 修复：无条件使用REINFORCE损失 =====
#         # Trainer会自动用 (Reward - Baseline) 加权这个损失
#         loss = -action_dist.log_prob(action)        
#         # Get chosen relation and time
#         batch_indices = torch.arange(next_relations.size(0), device=next_relations.device)
#         chosen_relation = next_relations[batch_indices, action]
#         chosen_time = next_times[batch_indices, action]
        
#         return loss, new_state, F.log_softmax(scores, dim=-1), action, chosen_relation, chosen_time

#     def forward(self, candidate_relation_sequence, candidate_entity_sequence, 
#                 candidate_time_sequence, current_entities, path_label, 
#                 query_relation, query_time, range_arr, first_step_of_test, T=3):
#         """
#         Unroll the agent for T steps
        
#         Returns:
#             all_loss: list of losses per step
#             all_logits: list of log probabilities per step
#             action_idx: list of actions taken per step
#         """
#         batch_size = query_relation.size(0)
#         device = query_relation.device
        
#         # Initialize
#         query_embedding = self.relation_lookup_table(query_relation)
        
#         # Initial LSTM state
#         h0 = torch.zeros(
#             self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
#             device=device
#         )
#         c0 = torch.zeros(
#             self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
#             device=device
#         )
#         state = (h0, c0)
        
#         # Dummy start
#         prev_relation = torch.ones(batch_size, dtype=torch.long, device=device) * self.dummy_start_label
#         prev_time = torch.zeros(batch_size, dtype=torch.long, device=device)
        
#         all_loss = []
#         all_logits = []
#         action_idx = []
        
#         for t in range(T):
#             next_possible_relations = candidate_relation_sequence[t]
#             next_possible_entities = candidate_entity_sequence[t]
#             next_possible_times = candidate_time_sequence[t]
#             current_entities_t = current_entities[t]
#             path_label_t = path_label[t]
            
#             loss, state, logits, idx, chosen_relation, chosen_time = self.step(
#                 next_possible_relations,
#                 next_possible_entities,
#                 next_possible_times,
#                 state,
#                 prev_relation,
#                 prev_time,
#                 query_embedding,
#                 query_time,
#                 current_entities_t,
#                 label_action=path_label_t,
#                 range_arr=range_arr,
#                 first_step_of_test=first_step_of_test
#             )
            
#             all_loss.append(loss)
#             all_logits.append(logits)
#             action_idx.append(idx)
            
#             prev_relation = chosen_relation
#             prev_time = chosen_time
        
#         return all_loss, all_logits, action_idx
# coding=UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, params):
        super(Agent, self).__init__()
        
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.time_vocab_size = len(params['time_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']
        
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        
        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        
        if self.use_entity_embeddings:
            self.m = 4 + 2
        else:
            self.m = 2 + 2
        
        # ===== Embedding tables =====
        if params['use_entity_embeddings']:
            self.entity_lookup_table = nn.Embedding(
                self.entity_vocab_size, 
                2 * self.entity_embedding_size
            )
            nn.init.xavier_normal_(self.entity_lookup_table.weight)
        else:
            self.entity_lookup_table = nn.Embedding(
                self.entity_vocab_size, 
                2 * self.entity_embedding_size
            )
            nn.init.zeros_(self.entity_lookup_table.weight)
        
        self.entity_lookup_table.weight.requires_grad = self.train_entities
        
        self.relation_lookup_table = nn.Embedding(
            self.action_vocab_size, 
            2 * self.embedding_size
        )
        nn.init.xavier_normal_(self.relation_lookup_table.weight)
        self.relation_lookup_table.weight.requires_grad = self.train_relations
        
        # ===== 🔥 相对时间编码器 (MLP) =====
        time_encoder_dim = 2 * self.entity_embedding_size
        self.time_diff_encoder = nn.Sequential(
            nn.Linear(1, time_encoder_dim),
            nn.Tanh(),  # 保留负数信息
            nn.Linear(time_encoder_dim, time_encoder_dim)
        )
        
        # 初始化MLP权重
        for layer in self.time_diff_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # ===== LSTM =====
        self.policy_step = nn.LSTM(
            input_size=self.m * self.embedding_size,
            hidden_size=self.m * self.hidden_size,
            num_layers=self.LSTM_Layers,
            batch_first=True
        )
        
        # ===== 🔥 修正: Policy MLP 输入维度 =====
        # 输入 = LSTM state + prev_entity (可选) + query_relation
        # 时间信息已通过候选动作编码传递,不需要单独输入
        if self.use_entity_embeddings:
            policy_input_dim = (self.m * self.hidden_size +       # LSTM state
                               2 * self.entity_embedding_size +    # prev_entity
                               2 * self.embedding_size)            # query_relation
        else:
            policy_input_dim = (self.m * self.hidden_size +       # LSTM state
                               2 * self.embedding_size)            # query_relation
        
        self.policy_mlp = nn.Linear(
            policy_input_dim,
            self.m * self.embedding_size
        )
        
        # Dummy start relation
        self.dummy_start_label = params['relation_vocab']['DUMMY_START_RELATION']
        # print(f"✅ Policy MLP input dim: {policy_input_dim}")
        # print(f"✅ Policy MLP output dim: {self.m * self.embedding_size}")
    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)
    
    def initialize_embeddings(self, entity_emb=None, relation_emb=None, time_emb=None):
        """Initialize embeddings from pretrained weights"""
        if entity_emb is not None:
            self.entity_lookup_table.weight.data.copy_(torch.from_numpy(entity_emb))
        if relation_emb is not None:
            self.relation_lookup_table.weight.data.copy_(torch.from_numpy(relation_emb))
        # time_emb参数被忽略，因为我们使用 time_diff_encoder
    
    # # def encode_action_with_relative_time(self, relations, entities, times, query_time):
    #     """
    #     统一的动作编码函数：使用相对时间差
        
    #     Args:
    #         relations: [batch_size, num_actions] 或 [batch_size, 1]
    #         entities: [batch_size, num_actions] 或 [batch_size, 1]
    #         times: [batch_size, num_actions] 或 [batch_size, 1] (绝对时间)
    #         query_time: [batch_size] (查询的绝对时间)
        
    #     Returns:
    #         action_embeddings: [batch_size, num_actions, emb_dim]
    #             其中 emb_dim = relation_dim + entity_dim + time_dim
    #     """
    #     # 1. 编码关系和实体
    #     relation_emb = self.relation_lookup_table(relations)
    #     entity_emb = self.entity_lookup_table(entities)
        
    #     # 2. 🔥 计算相对时间差 (query_time - action_time)
    #     # 正值 = 过去的动作, 负值 = 未来的动作
    #     query_time_expanded = query_time.unsqueeze(1).float()  # [B, 1]
    #     time_diffs = query_time_expanded - times.float()        # [B, num_actions]
        
    #     # 3. 🔥 归一化时间差到合理范围
    #     # 假设时间跨度为 time_vocab_size (如365天)
    #     time_diffs_normalized = time_diffs / self.time_vocab_size
        
    #     # 4. 🔥 通过MLP编码相对时间差
    #     time_diffs_expanded = time_diffs_normalized.unsqueeze(-1)  # [B, num_actions, 1]
    #     time_emb = self.time_diff_encoder(time_diffs_expanded)     # [B, num_actions, time_dim]
        
    #     # 5. 拼接所有特征
    #     if self.use_entity_embeddings:
    #         action_emb = torch.cat([relation_emb, entity_emb], dim=-1)
    #     else:
    #         action_emb = relation_emb
        
    #     # 最终动作表示 = [relation, entity (可选), relative_time]
    #     action_emb = torch.cat([action_emb, time_emb], dim=-1)
        
    #     return action_emb
    # def encode_action_with_relative_time(self, relations, entities, times, query_time):
    #     relation_emb = self.relation_lookup_table(relations)
    #     entity_emb = self.entity_lookup_table(entities)
        
    #     # 时间差编码
    #     query_time_expanded = query_time.unsqueeze(1).float()
    #     time_diffs = query_time_expanded - times.float()
        
    #     # 🔥 改用 log scale
    #     time_diffs_log = torch.sign(time_diffs) * torch.log1p(torch.abs(time_diffs))
    #     time_diffs_normalized = time_diffs_log / 5.0
        
    #     time_diffs_expanded = time_diffs_normalized.unsqueeze(-1)
    #     time_emb = self.time_diff_encoder(time_diffs_expanded)
        
    #     # 🔥 使用门控机制 (Sigmoid)
    #     time_gate = torch.sigmoid(time_emb)  # [B, max_actions, emb_dim]
        
    #     # 🔥 时间调制关系 (逐元素相乘)
    #     # 远期事件 → gate→0 → 关系信息衰减
    #     # 近期事件 → gate→1 → 关系信息保留
    #     relation_emb_gated = relation_emb * time_gate
        
    #     if self.use_entity_embeddings:
    #         action_emb = torch.cat([relation_emb_gated, entity_emb, time_emb], dim=-1)
    #     else:
    #         action_emb = torch.cat([relation_emb_gated, time_emb], dim=-1)
        
    #     return action_emb    
    def encode_action_with_relative_time(self, relations, entities, times, query_time):
        """改进版时间编码"""
        relation_emb = self.relation_lookup_table(relations)
        entity_emb = self.entity_lookup_table(entities)
        
        # 🔥 Log scale归一化
        query_time_expanded = query_time.unsqueeze(1).float()
        time_diffs = query_time_expanded - times.float()
        time_diffs_log = torch.sign(time_diffs) * torch.log1p(torch.abs(time_diffs))
        time_diffs_normalized = time_diffs_log / 3.0
        
        # MLP编码
        time_diffs_expanded = time_diffs_normalized.unsqueeze(-1)
        time_emb = self.time_diff_encoder(time_diffs_expanded)
        
        # 🔥 时间门控
        time_gate = torch.sigmoid(time_emb)
        relation_emb_gated = relation_emb * (0.5 + 0.5 * time_gate)
        
        # 拼接
        if self.use_entity_embeddings:
            action_emb = torch.cat([relation_emb_gated, entity_emb, time_emb], dim=-1)
        else:
            action_emb = torch.cat([relation_emb_gated, time_emb], dim=-1)
        
        return action_emb
    def step(self, next_relations, next_entities, next_times, prev_state, 
             prev_relation, prev_time, query_embedding, query_time,
             current_entities, label_action, range_arr, first_step_of_test):
        """
        执行单步推理
        
        关键逻辑:
        1. 用相对时间编码上一步动作和候选动作
        2. 时间信息完全融入动作表示,不需要单独处理query的时间
        3. Policy网络基于当前状态和query关系,选择最佳候选动作
        4. 硬惩罚未来动作 (time_diff < 0)
        """
        
        # ===== 1. 编码"上一步"动作 (使用相对时间) =====
        prev_action_embedding = self.encode_action_with_relative_time(
            prev_relation.unsqueeze(1),      # [B, 1]
            current_entities.unsqueeze(1),   # [B, 1]
            prev_time.unsqueeze(1),          # [B, 1]
            query_time                       # [B]
        )  # [B, 1, m*emb_dim]
        
        # LSTM step
        output, new_state = self.policy_step(prev_action_embedding, prev_state)
        output = output.squeeze(1)  # [B, m*hidden_dim]
        
        # Get current entity embedding
        prev_entity = self.entity_lookup_table(current_entities)  # [B, 2*entity_emb_dim]
        
        # Concatenate state components
        if self.use_entity_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output
        
        # ===== 2. 编码"候选"动作 (使用相对时间) =====
        # 🔥 关键: 时间信息已经编码到候选动作中
        candidate_action_embeddings = self.encode_action_with_relative_time(
            next_relations,  # [B, max_actions]
            next_entities,   # [B, max_actions]
            next_times,      # [B, max_actions]
            query_time       # [B]
        )  # [B, max_actions, m*emb_dim]
        
        # ===== 3. 🔥 修正: 直接拼接state和query_embedding =====
        # 不需要"融合"query和时间,因为:
        # - query_embedding 已经包含关系语义
        # - 时间信息通过候选动作的相对时间差体现
        # - Policy网络学习如何基于query选择合适的时间敏感动作
        state_query_concat = torch.cat([state, query_embedding], dim=-1)
        
        # Policy MLP: 生成action selection weights
        output = F.relu(self.policy_mlp(state_query_concat))  # [B, m*emb_dim]
        output_expanded = output.unsqueeze(1)                  # [B, 1, m*emb_dim]
        
        # Calculate scores: 内积相似度
        # 🔥 时间信息在candidate_action_embeddings中,自然参与打分
        prelim_scores = torch.sum(
            candidate_action_embeddings * output_expanded, 
            dim=2
        )  # [B, max_actions]
        
        # ===== 4. 🔥 硬惩罚未来动作 =====
        # 计算时间差 (正=过去, 负=未来)
        query_time_expanded = query_time.unsqueeze(1).float()
        time_diffs = query_time_expanded - next_times.float()
        
        # 未来动作 (time_diff < 0) 施加巨大负分
        future_penalty = -1e9 * (time_diffs < 0).float()
        prelim_scores = prelim_scores + future_penalty
        
        # Mask padding actions
        mask = (next_relations == self.rPAD)
        scores = prelim_scores.masked_fill(mask, -1e9)
        
        # Sample action from categorical distribution
        action_dist = torch.distributions.Categorical(logits=scores)
        action = action_dist.sample()
        
        # ===== 5. REINFORCE loss =====
        # Trainer会用 (Reward - Baseline) 加权这个损失
        loss = -action_dist.log_prob(action)
        
        # Get chosen relation and time
        batch_indices = torch.arange(next_relations.size(0), device=next_relations.device)
        chosen_relation = next_relations[batch_indices, action]
        chosen_time = next_times[batch_indices, action]  # 返回绝对时间
        
        return loss, new_state, F.log_softmax(scores, dim=-1), action, chosen_relation, chosen_time

    def forward(self, candidate_relation_sequence, candidate_entity_sequence, 
                candidate_time_sequence, current_entities, path_label, 
                query_relation, query_time, range_arr, first_step_of_test, T=3):
        """
        Unroll the agent for T steps
        
        Args:
            candidate_relation_sequence: list of [B, max_actions] for each timestep
            candidate_entity_sequence: list of [B, max_actions] for each timestep
            candidate_time_sequence: list of [B, max_actions] for each timestep
            current_entities: list of [B] for each timestep
            path_label: list of [B] (ground truth actions, unused in REINFORCE)
            query_relation: [B]
            query_time: [B]
            range_arr: range array for batch indexing
            first_step_of_test: bool
            T: number of steps
        
        Returns:
            all_loss: list of [B] losses per step
            all_logits: list of [B, max_actions] log probabilities per step
            action_idx: list of [B] actions taken per step
        """
        batch_size = query_relation.size(0)
        device = query_relation.device
        
        # Initialize query embedding
        query_embedding = self.relation_lookup_table(query_relation)
        
        # Initialize LSTM state
        h0 = torch.zeros(
            self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
            device=device
        )
        c0 = torch.zeros(
            self.LSTM_Layers, batch_size, self.m * self.hidden_size, 
            device=device
        )
        state = (h0, c0)
        # Dummy start action
        prev_relation = torch.ones(batch_size, dtype=torch.long, device=device) * self.dummy_start_label
        prev_time = torch.zeros(batch_size, dtype=torch.long, device=device)  # 时间=0作为起点
        
        all_loss = []
        all_logits = []
        action_idx = []
        
        # Unroll for T steps
        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            next_possible_times = candidate_time_sequence[t]
            current_entities_t = current_entities[t]
            path_label_t = path_label[t]  # Unused in REINFORCE
            
            loss, state, logits, idx, chosen_relation, chosen_time = self.step(
                next_possible_relations,
                next_possible_entities,
                next_possible_times,
                state,
                prev_relation,
                prev_time,
                query_embedding,
                query_time,
                current_entities_t,
                label_action=path_label_t,
                range_arr=range_arr,
                first_step_of_test=first_step_of_test
            )
            
            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            
            # Update for next step (传递绝对时间)
            prev_relation = chosen_relation
            prev_time = chosen_time
        
        return all_loss, all_logits, action_idx
