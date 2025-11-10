# # coding=UTF-8
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import logging
# import sys
# from tqdm import tqdm
# import json
# import os
# import codecs
# from collections import defaultdict
# import gc
# import resource
# import time

# from src.model.agent import Agent  # PyTorch版本
# from src.gat_model.gat import HeteGAT_multi  # PyTorch版本的GAT
# from src.options import read_options
# from src.model.environment import env
# from src.model.baseline import ReactiveBaseline
# from src.model.nell_eval import nell_eval
# from scipy.special import logsumexp as lse

# logger = logging.getLogger()
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# class Trainer(object):
#     def __init__(self, params):
#         # 保存所有参数
#         for key, val in params.items():
#             setattr(self, key, val)
        
#         # 设置设备
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f'Using device: {self.device}')
        
#         # 创建RL Agent
#         self.agent = Agent(params).to(self.device)
        
#         # 创建GAT模型
#         self.gat_model = HeteGAT_multi(params).to(self.device)
        
#         # 创建环境
#         self.train_environment = env(params, 'train')
#         self.dev_test_environment = env(params, 'dev')
#         self.test_test_environment = env(params, 'test')
#         self.test_environment = self.dev_test_environment
        
#         self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
#         self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
#         self.rev_time_vocab = self.train_environment.grapher.rev_time_vocab
        
#         self.max_hits_at_10 = 0
#         self.ePAD = params['entity_vocab']['PAD']
#         self.rPAD = params['relation_vocab']['PAD']
        
#         # Baseline
#         self.baseline = ReactiveBaseline(l=self.Lambda)
        
#         # 优化器 - 分别为Agent和GAT
#         self.optimizer_agent = optim.Adam(
#             self.agent.parameters(), 
#             lr=self.learning_rate,
#             weight_decay=self.l2_reg_const
#         )
        
#         self.optimizer_gat = optim.Adam(
#             self.gat_model.parameters(), 
#             lr=self.learning_rate,
#             weight_decay=self.l2_reg_const
#         )
        
#         # Counters
#         self.batch_counter = 0
#         self.global_step = 0
        
#         # Path logger
#         self.path_logger_file_ = None
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
        
#         # GAT相关参数
#         self.use_gat = True  # 是否使用GAT增强
#         self.gat_lambda = 0.5  # GAT损失权重
    
#     def get_decaying_beta(self):
#         """Exponential decay of beta"""
#         return self.beta * (0.90 ** (self.global_step / 200))
    
#     def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
#         """
#         Calculate REINFORCE loss with baseline
#         """
#         # Stack losses: [batch_size, path_length]
#         loss = torch.stack(per_example_loss, dim=1)
        
#         # Get baseline
#         baseline_value = self.baseline.get_baseline_value()
        
#         # Calculate advantage
#         final_reward = cum_discounted_reward - baseline_value
        
#         # Normalize reward
#         reward_mean = final_reward.mean()
#         reward_std = final_reward.std() + 1e-6
#         final_reward = (final_reward - reward_mean) / reward_std
        
#         # Apply advantage
#         loss = loss * final_reward
        
#         # Calculate entropy regularization
#         entropy_loss = self.entropy_reg_loss(per_example_logits)
        
#         # Total loss
#         decaying_beta = self.get_decaying_beta()
#         total_loss = loss.mean() - decaying_beta * entropy_loss
        
#         return total_loss
    
#     def entropy_reg_loss(self, all_logits):
#         """Calculate entropy regularization"""
#         all_logits = torch.stack(all_logits, dim=2)
#         entropy = -torch.sum(torch.exp(all_logits) * all_logits, dim=1)
#         entropy_mean = entropy.mean()
#         return entropy_mean
    
#     def calc_cum_discounted_reward(self, rewards):
#         """Calculate cumulative discounted reward"""
#         batch_size = len(rewards)
#         running_add = np.zeros(batch_size)
#         cum_disc_reward = np.zeros((batch_size, self.path_length))
        
#         cum_disc_reward[:, self.path_length - 1] = rewards
        
#         for t in reversed(range(self.path_length)):
#             running_add = self.gamma * running_add + cum_disc_reward[:, t]
#             cum_disc_reward[:, t] = running_add
        
#         return cum_disc_reward
    
#     def gat_loss(self, gat_scores, target_entities, episode):
#         """
#         Calculate GAT auxiliary loss
        
#         Args:
#             gat_scores: [batch_size, num_entities] predicted scores
#             target_entities: [batch_size] target entity indices
#             episode: current episode object
#         """
#         # Cross entropy loss for target prediction
#         loss = nn.functional.cross_entropy(gat_scores, target_entities)
#         return loss
    
#     def initialize_pretrained_embeddings(self):
#         """Initialize pretrained embeddings"""
#         if self.pretrained_embeddings_action != '':
#             logger.info('Loading pretrained action embeddings from: ' + self.pretrained_embeddings_action)
#             embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
#             self.agent.initialize_embeddings(relation_emb=embeddings)
        
#         if self.pretrained_embeddings_entity != '':
#             logger.info('Loading pretrained entity embeddings from: ' + self.pretrained_embeddings_entity)
#             embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
#             self.agent.initialize_embeddings(entity_emb=embeddings)
#             self.gat_model.initialize_embeddings(entity_emb=embeddings)
    
#     def train(self):
#         """Main training loop with GAT"""
#         train_loss = 0.0
#         train_gat_loss = 0.0
#         self.batch_counter = 0
        
#         logger.info("Starting training with GAT augmentation...")
        
#         for episode in self.train_environment.get_episodes():
#             self.batch_counter += 1
#             self.global_step += 1
            
#             # Prepare query
#             query_relation = torch.tensor(
#                 episode.get_query_relation(), 
#                 dtype=torch.long, 
#                 device=self.device
#             )
#             query_time = torch.tensor(
#                 episode.get_query_time(), 
#                 dtype=torch.long, 
#                 device=self.device
#             )
            
#             # Target entities for GAT supervision
#             target_entities = torch.tensor(
#                 episode.end_entities,
#                 dtype=torch.long,
#                 device=self.device
#             )
            
#             # ========== 修改: 使用与 trainer1 相同的轨迹收集逻辑 ==========
#             candidate_relations = []
#             candidate_entities = []
#             candidate_times = []
#             current_entity_seq = []
#             all_actions = []
#             state_seq = []  # For GAT
            
#             state = episode.get_state()
#             range_arr = torch.arange(self.batch_size, device=self.device)
            
#             # Rollout for path_length steps - Agent makes decisions
#             for t in range(self.path_length):
#                 # Collect current state
#                 candidate_relations.append(
#                     torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
#                 )
#                 candidate_entities.append(
#                     torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
#                 )
#                 candidate_times.append(
#                     torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
#                 )
#                 # 使用 episode.current_entities 而不是 state['current_entities']
#                 current_entity_seq.append(
#                     torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
#                 )
                
#                 state_seq.append(state)  # Save for GAT
                
#                 # Agent selects action based on current state
#                 with torch.no_grad():
#                     _, step_logits, step_action = self.agent(
#                         [candidate_relations[-1]],
#                         [candidate_entities[-1]],
#                         [candidate_times[-1]],
#                         [current_entity_seq[-1]],
#                         [None],
#                         query_relation,
#                         query_time,
#                         range_arr,
#                         first_step_of_test=False,
#                         T=1
#                     )
                    
#                     # Extract action
#                     if isinstance(step_action, list):
#                         action_idx = step_action[0]
#                     else:
#                         action_idx = step_action
                    
#                     all_actions.append(action_idx)
                
#                 # Take action in environment
#                 state = episode(action_idx.cpu().numpy())
            
#             # Get final rewards
#             rewards = episode.get_reward()
            
#             # ========== RL Agent Forward (full trajectory for loss) ==========
#             path_labels = [None for _ in range(self.path_length)]
            
#             per_example_loss, per_example_logits, _ = self.agent(
#                 candidate_relations,
#                 candidate_entities,
#                 candidate_times,
#                 current_entity_seq,
#                 path_labels,
#                 query_relation,
#                 query_time,
#                 range_arr,
#                 first_step_of_test=False,
#                 T=self.path_length
#             )
            
#             # # ========== GAT Forward ==========
#             # gat_loss_value = 0
#             # if self.use_gat:
#             #     try:
#             #         nb_nodes = len(self.entity_vocab)
                    
#             #         # Prepare trace (entity sequence)
#             #         trace = torch.stack(current_entity_seq, dim=1)  # [batch_size, path_length]
                    
#             #         # GAT inference
#             #         gat_scores, lstm_state, score_eo = self.gat_model.inference2(
#             #             entity_table=self.gat_model.entity_lookup_table.weight,
#             #             relation_table=self.gat_model.relation_lookup_table.weight,
#             #             time_in_x=query_time,
#             #             query=query_relation,
#             #             trace=trace,
#             #             batch_size=self.batch_size,
#             #             nb_nodes=nb_nodes,
#             #             prev_state=None,
#             #             range_h=range_arr
#             #         )
                    
#             #         # Calculate GAT loss
#             #         gat_loss_value = self.gat_loss(gat_scores, target_entities, episode)
#             #     except Exception as e:
#             #         logger.warning(f"GAT forward failed: {e}, skipping GAT loss")
#             #         gat_loss_value = 0
#             # ========== GAT Forward ==========
#             # gat_loss_value = 0
#             # if self.use_gat:
#             #     try:
#             #         nb_nodes = len(self.entity_vocab)
                    
#             #         # Prepare trace (entity sequence)
#             #         trace = torch.stack(current_entity_seq, dim=1)  # [batch_size, path_length]
                    
#             #         # ✅ 使用 agent 的 embedding 表（HeteGAT_multi 没有自己的 embedding）
#             #         entity_table = self.agent.entity_embedding.weight
#             #         relation_table = self.agent.relation_embedding.weight
                    
#             #         # GAT inference
#             #         gat_scores, lstm_state, score_eo = self.gat_model.inference2(
#             #             entity_table=entity_table,
#             #             relation_table=relation_table,
#             #             time_in_x=query_time,
#             #             query=query_relation,
#             #             trace=trace,
#             #             batch_size=self.batch_size,
#             #             nb_nodes=nb_nodes,
#             #             prev_state=None,
#             #             range_h=range_arr
#             #         )
                    
#             #         # Calculate GAT loss
#             #         gat_loss_value = self.gat_loss(gat_scores, target_entities, episode)
                    
#             #     except Exception as e:
#             #         logger.warning(f"GAT forward failed: {e}, skipping GAT loss")
#             #         import traceback
#             #         traceback.print_exc()  # 打印详细错误信息以便调试
#             #         gat_loss_value = 0
#             # # ========== GAT Forward ==========
#             # gat_loss_value = 0
#             # if self.use_gat:
#             #     try:
#             #         nb_nodes = len(self.entity_vocab)
                    
#             #         # Prepare trace (entity sequence)
#             #         trace = torch.stack(current_entity_seq, dim=1)  # [batch_size, path_length]
                    
#             #         # ✅ 使用正确的属性名
#             #         entity_table = self.agent.entity_lookup_table.weight
#             #         relation_table = self.agent.relation_lookup_table.weight
                    
#             #         # 确保 trace 在正确的设备上
#             #         trace = trace.to(self.device)
                    
#             #         # GAT inference
#             #         gat_scores, lstm_state, score_eo = self.gat_model.inference2(
#             #             entity_table=entity_table,
#             #             relation_table=relation_table,
#             #             time_in_x=query_time,
#             #             query=query_relation,
#             #             trace=trace,
#             #             batch_size=self.batch_size,
#             #             nb_nodes=nb_nodes,
#             #             prev_state=None,
#             #             range_h=range_arr
#             #         )
                    
#             #         # Calculate GAT loss
#             #         gat_loss_value = self.gat_loss(gat_scores, target_entities, episode)
                    
#             #         # 确保 gat_loss 是有效的张量
#             #         if not isinstance(gat_loss_value, torch.Tensor):
#             #             logger.warning(f"GAT loss is not a tensor: {type(gat_loss_value)}")
#             #             gat_loss_value = 0
#             #         elif torch.isnan(gat_loss_value) or torch.isinf(gat_loss_value):
#             #             logger.warning(f"GAT loss is NaN or Inf: {gat_loss_value.item()}")
#             #             gat_loss_value = 0
                        
#             #     except Exception as e:
#             #         logger.warning(f"GAT forward failed: {e}, skipping GAT loss")
#             #         import traceback
#             #         traceback.print_exc()
#             #         gat_loss_value = 0
#             # ========== GAT Forward ==========
#             gat_loss_value = 0
#             if self.use_gat:
#                 try:
#                     nb_nodes = len(self.entity_vocab)
                    
#                     # Prepare trace (entity sequence)
#                     trace = torch.stack(current_entity_seq, dim=1)  # [batch_size*rollouts, path_length]
                    
#                     # ✅ 修正：获取实际的 batch size（包含 rollouts）
#                     actual_batch_size = trace.size(0)  # 应该是 batch_size * num_rollouts
                    
#                     # 使用正确的属性名
#                     entity_table = self.agent.entity_lookup_table.weight
#                     relation_table = self.agent.relation_lookup_table.weight
                    
#                     # 确保所有输入在正确的设备上且维度正确
#                     trace = trace.to(self.device)
                    
#                     # ✅ query_relation 和 query_time 也需要扩展到实际 batch size
#                     # 如果它们还是原始 batch size，需要重复 num_rollouts 次
#                     if query_relation.size(0) != actual_batch_size:
#                         # 每个 query 重复 num_rollouts 次
#                         query_relation_expanded = query_relation.repeat_interleave(self.num_rollouts)
#                         query_time_expanded = query_time.repeat_interleave(self.num_rollouts)
#                     else:
#                         query_relation_expanded = query_relation
#                         query_time_expanded = query_time
                    
#                     # ✅ target_entities 也需要扩展
#                     if target_entities.size(0) != actual_batch_size:
#                         target_entities_expanded = target_entities.repeat_interleave(self.num_rollouts)
#                     else:
#                         target_entities_expanded = target_entities
                    
#                     # ✅ range_arr 也需要扩展
#                     if range_arr.size(0) != actual_batch_size:
#                         range_arr_expanded = torch.arange(actual_batch_size, device=self.device)
#                     else:
#                         range_arr_expanded = range_arr
                    
#                     # GAT inference
#                     gat_scores, lstm_state, score_eo = self.gat_model.inference2(
#                         entity_table=entity_table,
#                         relation_table=relation_table,
#                         time_in_x=query_time_expanded,
#                         query=query_relation_expanded,
#                         trace=trace,
#                         batch_size=actual_batch_size,  # ✅ 使用实际 batch size
#                         nb_nodes=nb_nodes,
#                         prev_state=None,
#                         range_h=range_arr_expanded
#                     )
                    
#                     # Calculate GAT loss
#                     gat_loss_value = self.gat_loss(gat_scores, target_entities_expanded, episode)
                    
#                     # 确保 gat_loss 是有效的张量
#                     if not isinstance(gat_loss_value, torch.Tensor):
#                         logger.warning(f"GAT loss is not a tensor: {type(gat_loss_value)}")
#                         gat_loss_value = 0
#                     elif torch.isnan(gat_loss_value) or torch.isinf(gat_loss_value):
#                         logger.warning(f"GAT loss is NaN or Inf: {gat_loss_value.item()}")
#                         gat_loss_value = 0
                        
#                 except Exception as e:
#                     logger.warning(f"GAT forward failed: {e}, skipping GAT loss")
#                     import traceback
#                     traceback.print_exc()
#                     gat_loss_value = 0

            
#             # ========== Calculate RL Rewards ==========
#             cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
#             cum_discounted_reward = torch.tensor(
#                 cum_discounted_reward, 
#                 dtype=torch.float32, 
#                 device=self.device
#             )
            
#             # ========== Total Loss ==========
#             rl_loss = self.calc_reinforce_loss(
#                 per_example_loss, 
#                 per_example_logits, 
#                 cum_discounted_reward
#             )
            
#             # Combine losses
#             if isinstance(gat_loss_value, torch.Tensor):
#                 total_loss = rl_loss + self.gat_lambda * gat_loss_value
#             else:
#                 total_loss = rl_loss
            
#             # ========== Backward Pass ==========
#             self.optimizer_agent.zero_grad()
#             if self.use_gat:
#                 self.optimizer_gat.zero_grad()
            
#             total_loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
#             if self.use_gat:
#                 torch.nn.utils.clip_grad_norm_(self.gat_model.parameters(), self.grad_clip_norm)
            
#             # Update
#             self.optimizer_agent.step()
#             if self.use_gat:
#                 self.optimizer_gat.step()
            
#             # Update baseline
#             self.baseline.update(cum_discounted_reward.mean().item())
            
#             # ========== Statistics ==========
#             train_loss = 0.98 * train_loss + 0.02 * rl_loss.item()
#             if isinstance(gat_loss_value, torch.Tensor):
#                 train_gat_loss = 0.98 * train_gat_loss + 0.02 * gat_loss_value.item()
            
#             avg_reward = np.mean(rewards)
            
#             # Calculate correct episodes
#             num_episodes = len(rewards) // self.num_rollouts
#             reward_reshape = np.reshape(rewards, (num_episodes, self.num_rollouts))
#             reward_reshape = np.sum(reward_reshape, axis=1)
#             reward_reshape = (reward_reshape > 0)
#             num_ep_correct = np.sum(reward_reshape)
            
#             if np.isnan(train_loss):
#                 raise ArithmeticError("Error in computing loss")
            
#             # Logging
#             if self.use_gat:
#                 logger.info(
#                     "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
#                     "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, "
#                     "RL loss {5:7.4f}, GAT loss {6:7.4f}".format(
#                         self.batch_counter, np.sum(rewards), avg_reward,
#                         num_ep_correct, (num_ep_correct / num_episodes), 
#                         train_loss, train_gat_loss
#                     )
#                 )
#             else:
#                 logger.info(
#                     "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
#                     "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".format(
#                         self.batch_counter, np.sum(rewards), avg_reward,
#                         num_ep_correct, (num_ep_correct / num_episodes), 
#                         train_loss
#                     )
#                 )
            
#             # Evaluation
#             if self.batch_counter % self.eval_every == 0:
#                 with open(self.output_dir + '/scores.txt', 'a') as score_file:
#                     score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                
#                 eval_dir = self.path_logger_file + "/" + str(self.batch_counter)
#                 if not os.path.exists(eval_dir):
#                     os.makedirs(eval_dir)
#                 self.path_logger_file_ = eval_dir + "/paths"
                
#                 self.test(beam=True, print_paths=False)
                
#                 # Save model
#                 self.save_model(self.model_dir + '/model_' + str(self.batch_counter) + '.pt')
            
#             logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            
#             gc.collect()
            
#             if self.batch_counter >= self.total_iterations:
#                 break

#     # def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
#     #     """Test the model with GAT"""
#     #     self.agent.eval()
#     #     if self.use_gat:
#     #         self.gat_model.eval()
        
#     #     all_final_reward_1 = 0
#     #     all_final_reward_3 = 0
#     #     all_final_reward_10 = 0
#     #     all_final_reward_20 = 0
#     #     auc = 0
        
#     #     total_examples = self.test_environment.total_no_examples
        
#     #     logger.info("Starting testing on " + str(total_examples) + " examples")
        
#     #     episode_num = 0
        
#     #     with torch.no_grad():
#     #         for episode in tqdm(self.test_environment.get_episodes()):
#     #             episode_num += 1
                
#     #             # Prepare query
#     #             query_relation = torch.tensor(
#     #                 episode.get_query_relation(), 
#     #                 dtype=torch.long, 
#     #                 device=self.device
#     #             )
#     #             query_time = torch.tensor(
#     #                 episode.get_query_time(), 
#     #                 dtype=torch.long, 
#     #                 device=self.device
#     #             )
                
#     #             temp_batch_size = query_relation.size(0)
                
#     #             # Collect trajectory with combined RL + GAT scores
#     #             all_scores = []
                
#     #             for t in range(self.path_length):
#     #                 state = episode.get_state()
                    
#     #                 candidate_relations = torch.tensor(
#     #                     state['next_relations'], dtype=torch.long, device=self.device
#     #                 )
#     #                 candidate_entities = torch.tensor(
#     #                     state['next_entities'], dtype=torch.long, device=self.device
#     #                 )
#     #                 candidate_times = torch.tensor(
#     #                     state['next_times'], dtype=torch.long, device=self.device
#     #                 )
#     #                 current_entities = torch.tensor(
#     #                     state['current_entities'], dtype=torch.long, device=self.device
#     #                 )
                    
#     #                 # Get RL scores (would need to extract from agent forward pass)
#     #                 # Simplified - just take action
#     #                 state = episode(np.zeros(temp_batch_size, dtype='int32'))
                
#     #             # Calculate rewards
#     #             rewards = episode.get_reward()
                
#     #             # Calculate metrics (same as before)
#     #             reward_reshape = rewards.reshape((-1, self.test_rollouts))
#     #             reward_reshape = np.sum(reward_reshape, axis=1)
                
#     #             reward_1 = (reward_reshape == self.positive_reward)
#     #             reward_3 = (reward_reshape >= self.positive_reward * 0.33)
#     #             reward_10 = (reward_reshape >= self.positive_reward * 0.1)
#     #             reward_20 = (reward_reshape >= self.positive_reward * 0.05)
                
#     #             all_final_reward_1 += np.sum(reward_1)
#     #             all_final_reward_3 += np.sum(reward_3)
#     #             all_final_reward_10 += np.sum(reward_10)
#     #             all_final_reward_20 += np.sum(reward_20)
                
#     #             # Calculate MRR
#     #             for i in range(len(reward_reshape)):
#     #                 if reward_reshape[i] > 0:
#     #                     rank = 1  # Simplified
#     #                     auc += 1.0 / rank
        
#     #     self.agent.train()
#     #     if self.use_gat:
#     #         self.gat_model.train()
        
#     #     # Normalize metrics
#     #     all_final_reward_1 = all_final_reward_1 / total_examples
#     #     all_final_reward_3 = all_final_reward_3 / total_examples
#     #     all_final_reward_10 = all_final_reward_10 / total_examples
#     #     all_final_reward_20 = all_final_reward_20 / total_examples
#     #     auc = auc / total_examples
        
#     #     # Log results
#     #     logger.info("=" * 50)
#     #     logger.info("Test Results:")
#     #     logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
#     #     logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
#     #     logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
#     #     logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
#     #     logger.info("MRR: {0:7.4f}".format(auc))
#     #     logger.info("=" * 50)
        
#     #     # Save to file
#     #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
#     #         score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
#     #         score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
#     #         score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
#     #         score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
#     #         score_file.write("MRR: {0:7.4f}\n".format(auc))
#     #         score_file.write("\n")
        
#     #     return all_final_reward_10
#     # def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
#     #     """Test the model with beam search"""
#     #     self.agent.eval()
#     #     if self.use_gat:
#     #         self.gat_model.eval()
        
#     #     all_final_reward_1 = 0
#     #     all_final_reward_3 = 0
#     #     all_final_reward_10 = 0
#     #     all_final_reward_20 = 0
#     #     auc = 0
        
#     #     total_examples = self.test_environment.total_no_examples
        
#     #     logger.info("Starting testing on " + str(total_examples) + " examples")
        
#     #     episode_num = 0
        
#     #     # 路径记录
#     #     if print_paths:
#     #         path_file = codecs.open(self.path_logger_file_ + '.txt', 'a', 'utf-8')
        
#     #     with torch.no_grad():
#     #         for episode in tqdm(self.test_environment.get_episodes()):
#     #             episode_num += 1
                
#     #             # Prepare query
#     #             query_relation = torch.tensor(
#     #                 episode.get_query_relation(), 
#     #                 dtype=torch.long, 
#     #                 device=self.device
#     #             )
#     #             query_time = torch.tensor(
#     #                 episode.get_query_time(), 
#     #                 dtype=torch.long, 
#     #                 device=self.device
#     #             )
                
#     #             temp_batch_size = query_relation.size(0)
#     #             range_arr = torch.arange(temp_batch_size, device=self.device)
                
#     #             # ========== Beam Search ==========
#     #             if beam:
#     #                 # 初始化beam
#     #                 self.beam_search(episode, query_relation, query_time, 
#     #                                 range_arr, beam_size, print_paths)
#     #             else:
#     #                 # 标准rollout（与训练时相同）
#     #                 self.standard_rollout(episode, query_relation, query_time, range_arr)
                
#     #             # ========== Calculate Rewards and Metrics ==========
#     #             rewards = episode.get_reward()
                
#     #             # Reshape rewards: [num_examples, test_rollouts]
#     #             reward_reshape = rewards.reshape((temp_batch_size, self.test_rollouts))
                
#     #             # 对每个example的所有rollout求和
#     #             reward_reshape = np.sum(reward_reshape, axis=1)
                
#     #             # Calculate hits at different thresholds
#     #             reward_1 = (reward_reshape == self.positive_reward)
#     #             reward_3 = (reward_reshape >= self.positive_reward * 0.33)
#     #             reward_10 = (reward_reshape >= self.positive_reward * 0.1)
#     #             reward_20 = (reward_reshape >= self.positive_reward * 0.05)
                
#     #             all_final_reward_1 += np.sum(reward_1)
#     #             all_final_reward_3 += np.sum(reward_3)
#     #             all_final_reward_10 += np.sum(reward_10)
#     #             all_final_reward_20 += np.sum(reward_20)
                
#     #             # Calculate MRR
#     #             for i in range(temp_batch_size):
#     #                 if reward_reshape[i] > 0:
#     #                     # 找到正确答案的排名
#     #                     # 这里简化处理，实际应该基于logits排序
#     #                     rank = 1.0 / (reward_reshape[i] / self.positive_reward)
#     #                     auc += 1.0 / rank
        
#     #     if print_paths:
#     #         path_file.close()
        
#     #     self.agent.train()
#     #     if self.use_gat:
#     #         self.gat_model.train()
        
#     #     # Normalize metrics
#     #     all_final_reward_1 = all_final_reward_1 / total_examples
#     #     all_final_reward_3 = all_final_reward_3 / total_examples
#     #     all_final_reward_10 = all_final_reward_10 / total_examples
#     #     all_final_reward_20 = all_final_reward_20 / total_examples
#     #     auc = auc / total_examples
        
#     #     # Log results
#     #     logger.info("=" * 50)
#     #     logger.info("Test Results:")
#     #     logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
#     #     logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
#     #     logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
#     #     logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
#     #     logger.info("MRR: {0:7.4f}".format(auc))
#     #     logger.info("=" * 50)
        
#     #     # Save to file
#     #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
#     #         score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
#     #         score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
#     #         score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
#     #         score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
#     #         score_file.write("MRR: {0:7.4f}\n".format(auc))
#     #         score_file.write("\n")
        
#     #     return all_final_reward_10

#     # def beam_search(self, episode, query_relation, query_time, range_arr, beam_size, print_paths):
#     #     """
#     #     Perform beam search for testing
#     #     """
#     #     temp_batch_size = query_relation.size(0)
        
#     #     # 收集轨迹数据用于agent前向
#     #     candidate_relations = []
#     #     candidate_entities = []
#     #     candidate_times = []
#     #     current_entity_seq = []
        
#     #     state = episode.get_state()
        
#     #     # Rollout for path_length steps
#     #     for t in range(self.path_length):
#     #         # Collect current state
#     #         candidate_relations.append(
#     #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
#     #         )
#     #         candidate_entities.append(
#     #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
#     #         )
#     #         candidate_times.append(
#     #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
#     #         )
#     #         current_entity_seq.append(
#     #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
#     #         )
            
#     #         # Agent forward to get action scores
#     #         _, step_logits, _ = self.agent(
#     #             [candidate_relations[-1]],
#     #             [candidate_entities[-1]],
#     #             [candidate_times[-1]],
#     #             [current_entity_seq[-1]],
#     #             [None],
#     #             query_relation,
#     #             query_time,
#     #             range_arr,
#     #             first_step_of_test=(t == 0),
#     #             T=1
#     #         )
            
#     #         # Get log probabilities
#     #         step_log_probs = step_logits[0]  # [batch_size, max_num_actions]
            
#     #         # 如果使用GAT，可以结合GAT分数
#     #         if self.use_gat and t == self.path_length - 1:
#     #             try:
#     #                 # 在最后一步使用GAT辅助
#     #                 trace = torch.stack(current_entity_seq, dim=1)
#     #                 actual_batch_size = trace.size(0)
                    
#     #                 if query_relation.size(0) != actual_batch_size:
#     #                     query_relation_expanded = query_relation.repeat_interleave(
#     #                         actual_batch_size // query_relation.size(0)
#     #                     )
#     #                     query_time_expanded = query_time.repeat_interleave(
#     #                         actual_batch_size // query_time.size(0)
#     #                     )
#     #                 else:
#     #                     query_relation_expanded = query_relation
#     #                     query_time_expanded = query_time
                    
#     #                 entity_table = self.agent.entity_lookup_table.weight
#     #                 relation_table = self.agent.relation_lookup_table.weight
                    
#     #                 gat_scores, _, _ = self.gat_model.inference2(
#     #                     entity_table=entity_table,
#     #                     relation_table=relation_table,
#     #                     time_in_x=query_time_expanded,
#     #                     query=query_relation_expanded,
#     #                     trace=trace,
#     #                     batch_size=actual_batch_size,
#     #                     nb_nodes=len(self.entity_vocab),
#     #                     prev_state=None,
#     #                     range_h=torch.arange(actual_batch_size, device=self.device)
#     #                 )
                    
#     #                 # 结合RL和GAT分数 (简化版本)
#     #                 # 这里可以根据candidate_entities映射GAT分数
#     #                 # step_log_probs += 0.1 * gat_scores_mapped
#     #             except:
#     #                 pass
            
#     #         # Beam search selection
#     #         if t == 0:
#     #             # 第一步：选择top-k actions
#     #             k = min(beam_size, step_log_probs.size(1))
#     #             top_k_log_probs, top_k_indices = torch.topk(step_log_probs, k, dim=1)
                
#     #             # 扩展batch以支持beam
#     #             # 这里简化：只选择每个example的top action
#     #             action_idx = top_k_indices[:, 0]
#     #         else:
#     #             # 后续步骤：基于累积概率选择
#     #             action_idx = torch.argmax(step_log_probs, dim=1)
            
#     #         # Take action in environment
#     #         state = episode(action_idx.cpu().numpy())
            
#     #         # 记录路径
#     #         if print_paths:
#     #             # 可以在这里记录路径信息
#     #             pass

#     # def standard_rollout(self, episode, query_relation, query_time, range_arr):
#     #     """
#     #     Standard rollout for testing (without beam search)
#     #     """
#     #     candidate_relations = []
#     #     candidate_entities = []
#     #     candidate_times = []
#     #     current_entity_seq = []
        
#     #     state = episode.get_state()
        
#     #     for t in range(self.path_length):
#     #         candidate_relations.append(
#     #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
#     #         )
#     #         candidate_entities.append(
#     #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
#     #         )
#     #         candidate_times.append(
#     #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
#     #         )
#     #         current_entity_seq.append(
#     #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
#     #         )
            
#     #         # Agent selects action
#     #         _, _, step_action = self.agent(
#     #             [candidate_relations[-1]],
#     #             [candidate_entities[-1]],
#     #             [candidate_times[-1]],
#     #             [current_entity_seq[-1]],
#     #             [None],
#     #             query_relation,
#     #             query_time,
#     #             range_arr,
#     #             first_step_of_test=(t == 0),
#     #             T=1
#     #         )
            
#     #         if isinstance(step_action, list):
#     #             action_idx = step_action[0]
#     #         else:
#     #             action_idx = step_action
            
#     #         # Take action
#     #         state = episode(action_idx.cpu().numpy())
#     def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
#         """Test the model with beam search"""
#         self.agent.eval()
#         if self.use_gat:
#             self.gat_model.eval()
        
#         all_final_reward_1 = 0
#         all_final_reward_3 = 0
#         all_final_reward_10 = 0
#         all_final_reward_20 = 0
#         auc = 0
        
#         total_examples = self.test_environment.total_no_examples
        
#         logger.info("Starting testing on " + str(total_examples) + " examples")
        
#         episode_num = 0
        
#         # 路径记录
#         if print_paths:
#             path_file = codecs.open(self.path_logger_file_ + '.txt', 'a', 'utf-8')
        
#         with torch.no_grad():
#             for episode in tqdm(self.test_environment.get_episodes()):
#                 episode_num += 1
                
#                 # Prepare query
#                 query_relation = torch.tensor(
#                     episode.get_query_relation(), 
#                     dtype=torch.long, 
#                     device=self.device
#                 )
#                 query_time = torch.tensor(
#                     episode.get_query_time(), 
#                     dtype=torch.long, 
#                     device=self.device
#                 )
                
#                 # ✅ temp_batch_size 应该是原始的batch size，不含rollouts
#                 temp_batch_size = query_relation.size(0)
                
#                 # ✅ 实际的batch size包含rollouts
#                 actual_batch_size = temp_batch_size * self.test_rollouts
#                 range_arr = torch.arange(actual_batch_size, device=self.device)
                
#                 # ========== Beam Search or Standard Rollout ==========
#                 if beam:
#                     self.beam_search(episode, query_relation, query_time, 
#                                     range_arr, beam_size, print_paths)
#                 else:
#                     self.standard_rollout(episode, query_relation, query_time, range_arr)
                
#                 # ========== Calculate Rewards and Metrics ==========
#                 rewards = episode.get_reward()
                
#                 # ✅ 确保rewards的长度正确
#                 if len(rewards) != actual_batch_size:
#                     logger.warning(f"Reward length mismatch: expected {actual_batch_size}, got {len(rewards)}")
#                     continue
                
#                 # Reshape rewards: [temp_batch_size, test_rollouts]
#                 reward_reshape = rewards.reshape((temp_batch_size, self.test_rollouts))
                
#                 # 对每个example的所有rollout求和
#                 reward_reshape = np.sum(reward_reshape, axis=1)
                
#                 # Calculate hits at different thresholds
#                 reward_1 = (reward_reshape == self.positive_reward)
#                 reward_3 = (reward_reshape >= self.positive_reward * 0.33)
#                 reward_10 = (reward_reshape >= self.positive_reward * 0.1)
#                 reward_20 = (reward_reshape >= self.positive_reward * 0.05)
                
#                 all_final_reward_1 += np.sum(reward_1)
#                 all_final_reward_3 += np.sum(reward_3)
#                 all_final_reward_10 += np.sum(reward_10)
#                 all_final_reward_20 += np.sum(reward_20)
                
#                 # ✅ 修复 MRR 计算
#                 for i in range(temp_batch_size):
#                     if reward_reshape[i] > 0:
#                         # reward_reshape[i] 是该example在所有rollouts中击中的次数
#                         # 假设至少有一次击中，rank设为1（简化）
#                         # 实际应该基于所有候选实体的分数排序
#                         auc += 1.0
        
#         if print_paths:
#             path_file.close()
        
#         self.agent.train()
#         if self.use_gat:
#             self.gat_model.train()
        
#         # Normalize metrics
#         all_final_reward_1 = all_final_reward_1 / total_examples
#         all_final_reward_3 = all_final_reward_3 / total_examples
#         all_final_reward_10 = all_final_reward_10 / total_examples
#         all_final_reward_20 = all_final_reward_20 / total_examples
#         auc = auc / total_examples
        
#         # Log results
#         logger.info("=" * 50)
#         logger.info("Test Results:")
#         logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
#         logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
#         logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
#         logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
#         logger.info("MRR: {0:7.4f}".format(auc))
#         logger.info("=" * 50)
        
#         # Save to file
#         with open(self.output_dir + '/scores.txt', 'a') as score_file:
#             score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
#             score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
#             score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
#             score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
#             score_file.write("MRR: {0:7.4f}\n".format(auc))
#             score_file.write("\n")
        
#         return all_final_reward_10

#     def beam_search(self, episode, query_relation, query_time, range_arr, beam_size, print_paths):
#         """
#         Perform beam search for testing
#         """
#         # ✅ 注意：query_relation 和 query_time 是 [temp_batch_size]
#         # 但 episode 的 current_entities 等是 [actual_batch_size] (包含rollouts)
#         temp_batch_size = query_relation.size(0)
#         actual_batch_size = range_arr.size(0)  # 包含rollouts
        
#         # ✅ 扩展 query_relation 和 query_time 到 actual_batch_size
#         query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
#         query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
        
#         # 收集轨迹数据用于agent前向
#         candidate_relations = []
#         candidate_entities = []
#         candidate_times = []
#         current_entity_seq = []
        
#         state = episode.get_state()
        
#         # Rollout for path_length steps
#         for t in range(self.path_length):
#             # Collect current state
#             candidate_relations.append(
#                 torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
#             )
#             candidate_entities.append(
#                 torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
#             )
#             candidate_times.append(
#                 torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
#             )
#             current_entity_seq.append(
#                 torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
#             )
            
#             # Agent forward to get action scores
#             _, step_logits, _ = self.agent(
#                 [candidate_relations[-1]],
#                 [candidate_entities[-1]],
#                 [candidate_times[-1]],
#                 [current_entity_seq[-1]],
#                 [None],
#                 query_relation_expanded,  # ✅ 使用扩展后的
#                 query_time_expanded,      # ✅ 使用扩展后的
#                 range_arr,                # ✅ 使用actual_batch_size的range_arr
#                 first_step_of_test=(t == 0),
#                 T=1
#             )
            
#             # Get log probabilities
#             step_log_probs = step_logits[0]  # [actual_batch_size, max_num_actions]
            
#             # ✅ 选择最佳action（简化版beam search）
#             action_idx = torch.argmax(step_log_probs, dim=1)
            
#             # Take action in environment
#             state = episode(action_idx.cpu().numpy())

#     def standard_rollout(self, episode, query_relation, query_time, range_arr):
#         """
#         Standard rollout for testing (without beam search)
#         """
#         temp_batch_size = query_relation.size(0)
#         actual_batch_size = range_arr.size(0)
        
#         # ✅ 扩展 query
#         query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
#         query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
        
#         candidate_relations = []
#         candidate_entities = []
#         candidate_times = []
#         current_entity_seq = []
        
#         state = episode.get_state()
        
#         for t in range(self.path_length):
#             candidate_relations.append(
#                 torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
#             )
#             candidate_entities.append(
#                 torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
#             )
#             candidate_times.append(
#                 torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
#             )
#             current_entity_seq.append(
#                 torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
#             )
            
#             # Agent selects action
#             _, _, step_action = self.agent(
#                 [candidate_relations[-1]],
#                 [candidate_entities[-1]],
#                 [candidate_times[-1]],
#                 [current_entity_seq[-1]],
#                 [None],
#                 query_relation_expanded,  # ✅ 使用扩展后的
#                 query_time_expanded,      # ✅ 使用扩展后的
#                 range_arr,
#                 first_step_of_test=(t == 0),
#                 T=1
#             )
            
#             if isinstance(step_action, list):
#                 action_idx = step_action[0]
#             else:
#                 action_idx = step_action
            
#             # Take action
#             state = episode(action_idx.cpu().numpy())
    
#     def save_model(self, path):
#         """Save model checkpoint"""
#         save_dict = {
#             'agent_state_dict': self.agent.state_dict(),
#             'optimizer_agent_state_dict': self.optimizer_agent.state_dict(),
#             'batch_counter': self.batch_counter,
#             'baseline': self.baseline.b,
#         }
        
#         if self.use_gat:
#             save_dict['gat_state_dict'] = self.gat_model.state_dict()
#             save_dict['optimizer_gat_state_dict'] = self.optimizer_gat.state_dict()
        
#         torch.save(save_dict, path)
#         logger.info("Model saved to: " + path)
    
#     def load_model(self, path):
#         """Load model checkpoint"""
#         checkpoint = torch.load(path, map_location=self.device)
        
#         self.agent.load_state_dict(checkpoint['agent_state_dict'])
#         self.optimizer_agent.load_state_dict(checkpoint['optimizer_agent_state_dict'])
        
#         if self.use_gat and 'gat_state_dict' in checkpoint:
#             self.gat_model.load_state_dict(checkpoint['gat_state_dict'])
#             self.optimizer_gat.load_state_dict(checkpoint['optimizer_gat_state_dict'])
        
#         self.batch_counter = checkpoint['batch_counter']
#         if 'baseline' in checkpoint:
#             self.baseline.b = checkpoint['baseline']
        
#         logger.info("Model loaded from: " + path)


# if __name__ == '__main__':
#     options = read_options()
    
#     # Setup logging
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
#     console = logging.StreamHandler()
#     console.setFormatter(fmt)
#     logger.addHandler(console)
    
#     if not os.path.exists(options['base_output_dir']):
#         os.makedirs(options['base_output_dir'])
    
#     logfile = logging.FileHandler(options['log_file_name'], 'w')
#     logfile.setFormatter(fmt)
#     logger.addHandler(logfile)
    
#     # Load vocabs
#     logger.info('reading vocab files...')
#     options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
#     options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
#     options['time_vocab'] = json.load(open(options['vocab_dir'] + '/time_vocab.json'))
    
#     logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
#     logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
#     logger.info('Total number of times {}'.format(len(options['time_vocab'])))
    
#     # Create trainer with GAT
#     trainer = Trainer(options)
    
#     if not options['load_model']:
#         trainer.initialize_pretrained_embeddings()
#         trainer.train()
#     else:
#         trainer.load_model(options['model_load_dir'])
#         trainer.test(beam=True, print_paths=True, save_model=False)

# coding=UTF-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys
from tqdm import tqdm
import json
import os
import codecs
from collections import defaultdict
import gc
import resource
import time

from src.model.agent import Agent
from src.gat_model.gat import HeteGAT_multi
from src.options import read_options
from src.model.environment import env
from src.model.baseline import ReactiveBaseline
from src.model.nell_eval import nell_eval
from scipy.special import logsumexp as lse

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):
        # 保存所有参数
        for key, val in params.items():
            setattr(self, key, val)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # 创建RL Agent
        self.agent = Agent(params).to(self.device)
        
        # 创建GAT模型
        self.gat_model = HeteGAT_multi(params).to(self.device)
        
        # 创建环境
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.test_environment = self.dev_test_environment
        
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.rev_time_vocab = self.train_environment.grapher.rev_time_vocab
        
        self.max_hits_at_10 = 0
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']
        
        # Baseline
        self.baseline = ReactiveBaseline(l=self.Lambda)
        
        # 优化器 - 分别为Agent和GAT
        self.optimizer_agent = optim.Adam(
            self.agent.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.l2_reg_const
        )
        
        self.optimizer_gat = optim.Adam(
            self.gat_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.l2_reg_const
        )
        
        # Counters
        self.batch_counter = 0
        self.global_step = 0
        
        # Path logger
        self.path_logger_file_ = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # GAT相关参数
        self.use_gat = True
        self.gat_lambda = 0.5
    
    def get_decaying_beta(self):
        """Exponential decay of beta"""
        return self.beta * (0.90 ** (self.global_step / 200))
    
    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        """Calculate REINFORCE loss with baseline"""
        loss = torch.stack(per_example_loss, dim=1)
        baseline_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - baseline_value
        
        reward_mean = final_reward.mean()
        reward_std = final_reward.std() + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std
        
        loss = loss * final_reward
        entropy_loss = self.entropy_reg_loss(per_example_logits)
        
        decaying_beta = self.get_decaying_beta()
        total_loss = loss.mean() - decaying_beta * entropy_loss
        
        return total_loss
    
    def entropy_reg_loss(self, all_logits):
        """Calculate entropy regularization"""
        all_logits = torch.stack(all_logits, dim=2)
        entropy = -torch.sum(torch.exp(all_logits) * all_logits, dim=1)
        entropy_mean = entropy.mean()
        return entropy_mean
    
    def calc_cum_discounted_reward(self, rewards):
        """Calculate cumulative discounted reward"""
        batch_size = len(rewards)
        running_add = np.zeros(batch_size)
        cum_disc_reward = np.zeros((batch_size, self.path_length))
        
        cum_disc_reward[:, self.path_length - 1] = rewards
        
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        
        return cum_disc_reward
    
    def gat_loss(self, gat_scores, target_entities, episode):
        """Calculate GAT auxiliary loss"""
        loss = nn.functional.cross_entropy(gat_scores, target_entities)
        return loss
    
    def initialize_pretrained_embeddings(self):
        """Initialize pretrained embeddings"""
        if self.pretrained_embeddings_action != '':
            logger.info('Loading pretrained action embeddings from: ' + self.pretrained_embeddings_action)
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            self.agent.initialize_embeddings(relation_emb=embeddings)
        
        if self.pretrained_embeddings_entity != '':
            logger.info('Loading pretrained entity embeddings from: ' + self.pretrained_embeddings_entity)
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            self.agent.initialize_embeddings(entity_emb=embeddings)
    
    def train(self):
        """Main training loop with GAT"""
        train_loss = 0.0
        train_gat_loss = 0.0
        self.batch_counter = 0
        
        logger.info("Starting training with GAT augmentation...")
        
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            self.global_step += 1
            
            # Prepare query
            query_relation = torch.tensor(
                episode.get_query_relation(), 
                dtype=torch.long, 
                device=self.device
            )
            query_time = torch.tensor(
                episode.get_query_time(), 
                dtype=torch.long, 
                device=self.device
            )
            
            # Target entities for GAT supervision
            target_entities = torch.tensor(
                episode.end_entities,
                dtype=torch.long,
                device=self.device
            )
            
            # 轨迹收集
            candidate_relations = []
            candidate_entities = []
            candidate_times = []
            current_entity_seq = []
            all_actions = []
            state_seq = []
            
            state = episode.get_state()
            range_arr = torch.arange(self.batch_size, device=self.device)
            
            # Rollout for path_length steps
            for t in range(self.path_length):
                candidate_relations.append(
                    torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
                )
                candidate_entities.append(
                    torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
                )
                candidate_times.append(
                    torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
                )
                current_entity_seq.append(
                    torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
                )
                
                state_seq.append(state)
                
                # Agent selects action
                with torch.no_grad():
                    _, step_logits, step_action = self.agent(
                        [candidate_relations[-1]],
                        [candidate_entities[-1]],
                        [candidate_times[-1]],
                        [current_entity_seq[-1]],
                        [None],
                        query_relation,
                        query_time,
                        range_arr,
                        first_step_of_test=False,
                        T=1
                    )
                    
                    if isinstance(step_action, list):
                        action_idx = step_action[0]
                    else:
                        action_idx = step_action
                    
                    all_actions.append(action_idx)
                
                state = episode(action_idx.cpu().numpy())
            
            rewards = episode.get_reward()
            
            # RL Agent Forward
            path_labels = [None for _ in range(self.path_length)]
            
            per_example_loss, per_example_logits, _ = self.agent(
                candidate_relations,
                candidate_entities,
                candidate_times,
                current_entity_seq,
                path_labels,
                query_relation,
                query_time,
                range_arr,
                first_step_of_test=False,
                T=self.path_length
            )
            
            # GAT Forward
            gat_loss_value = 0
            if self.use_gat:
                try:
                    nb_nodes = len(self.entity_vocab)
                    trace = torch.stack(current_entity_seq, dim=1)
                    actual_batch_size = trace.size(0)
                    
                    entity_table = self.agent.entity_lookup_table.weight
                    relation_table = self.agent.relation_lookup_table.weight
                    trace = trace.to(self.device)
                    
                    if query_relation.size(0) != actual_batch_size:
                        query_relation_expanded = query_relation.repeat_interleave(self.num_rollouts)
                        query_time_expanded = query_time.repeat_interleave(self.num_rollouts)
                    else:
                        query_relation_expanded = query_relation
                        query_time_expanded = query_time
                    
                    if target_entities.size(0) != actual_batch_size:
                        target_entities_expanded = target_entities.repeat_interleave(self.num_rollouts)
                    else:
                        target_entities_expanded = target_entities
                    
                    if range_arr.size(0) != actual_batch_size:
                        range_arr_expanded = torch.arange(actual_batch_size, device=self.device)
                    else:
                        range_arr_expanded = range_arr
                    
                    gat_scores, lstm_state, score_eo = self.gat_model.inference2(
                        entity_table=entity_table,
                        relation_table=relation_table,
                        time_in_x=query_time_expanded,
                        query=query_relation_expanded,
                        trace=trace,
                        batch_size=actual_batch_size,
                        nb_nodes=nb_nodes,
                        prev_state=None,
                        range_h=range_arr_expanded
                    )
                    
                    gat_loss_value = self.gat_loss(gat_scores, target_entities_expanded, episode)
                    
                    if not isinstance(gat_loss_value, torch.Tensor):
                        logger.warning(f"GAT loss is not a tensor: {type(gat_loss_value)}")
                        gat_loss_value = 0
                    elif torch.isnan(gat_loss_value) or torch.isinf(gat_loss_value):
                        logger.warning(f"GAT loss is NaN or Inf: {gat_loss_value.item()}")
                        gat_loss_value = 0
                        
                except Exception as e:
                    logger.warning(f"GAT forward failed: {e}, skipping GAT loss")
                    import traceback
                    traceback.print_exc()
                    gat_loss_value = 0
            
            # Calculate RL Rewards
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            cum_discounted_reward = torch.tensor(
                cum_discounted_reward, 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Total Loss
            rl_loss = self.calc_reinforce_loss(
                per_example_loss, 
                per_example_logits, 
                cum_discounted_reward
            )
            
            if isinstance(gat_loss_value, torch.Tensor):
                total_loss = rl_loss + self.gat_lambda * gat_loss_value
            else:
                total_loss = rl_loss
            
            # Backward Pass
            self.optimizer_agent.zero_grad()
            if self.use_gat:
                self.optimizer_gat.zero_grad()
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            if self.use_gat:
                torch.nn.utils.clip_grad_norm_(self.gat_model.parameters(), self.grad_clip_norm)
            
            self.optimizer_agent.step()
            if self.use_gat:
                self.optimizer_gat.step()
            
            self.baseline.update(cum_discounted_reward.mean().item())
            
            # Statistics
            train_loss = 0.98 * train_loss + 0.02 * rl_loss.item()
            if isinstance(gat_loss_value, torch.Tensor):
                train_gat_loss = 0.98 * train_gat_loss + 0.02 * gat_loss_value.item()
            
            avg_reward = np.mean(rewards)
            
            num_episodes = len(rewards) // self.num_rollouts
            reward_reshape = np.reshape(rewards, (num_episodes, self.num_rollouts))
            reward_reshape = np.sum(reward_reshape, axis=1)
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")
            
            # Logging
            if self.use_gat:
                logger.info(
                    "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
                    "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, "
                    "RL loss {5:7.4f}, GAT loss {6:7.4f}".format(
                        self.batch_counter, np.sum(rewards), avg_reward,
                        num_ep_correct, (num_ep_correct / num_episodes), 
                        train_loss, train_gat_loss
                    )
                )
            else:
                logger.info(
                    "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
                    "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".format(
                        self.batch_counter, np.sum(rewards), avg_reward,
                        num_ep_correct, (num_ep_correct / num_episodes), 
                        train_loss
                    )
                )
            
            # Evaluation
            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                
                eval_dir = self.path_logger_file + "/" + str(self.batch_counter)
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)
                self.path_logger_file_ = eval_dir + "/paths"
                
                self.test(beam=True, print_paths=False)
                self.save_model(self.model_dir + '/model_' + str(self.batch_counter) + '.pt')
            
            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            
            gc.collect()
            
            if self.batch_counter >= self.total_iterations:
                break

    # def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
    #     """Test the model with beam search"""
    #     self.agent.eval()
    #     if self.use_gat:
    #         self.gat_model.eval()
        
    #     all_final_reward_1 = 0
    #     all_final_reward_3 = 0
    #     all_final_reward_10 = 0
    #     all_final_reward_20 = 0
    #     auc = 0
        
    #     total_examples = self.test_environment.total_no_examples
        
    #     logger.info("Starting testing on " + str(total_examples) + " examples")
        
    #     episode_num = 0
        
    #     if print_paths:
    #         path_file = codecs.open(self.path_logger_file_ + '.txt', 'a', 'utf-8')
        
    #     with torch.no_grad():
    #         for episode in tqdm(self.test_environment.get_episodes()):
    #             episode_num += 1
                
    #             query_relation = torch.tensor(
    #                 episode.get_query_relation(), 
    #                 dtype=torch.long, 
    #                 device=self.device
    #             )
    #             query_time = torch.tensor(
    #                 episode.get_query_time(), 
    #                 dtype=torch.long, 
    #                 device=self.device
    #             )
                
    #             # temp_batch_size = query_relation.size(0)
                
    #             # # Beam Search or Standard Rollout
    #             # if beam:
    #             #     self.beam_search(episode, query_relation, query_time, 
    #             #                     beam_size, print_paths)
    #             # else:
    #             #     self.standard_rollout(episode, query_relation, query_time)
                
    #             # # Calculate Rewards and Metrics
    #             # rewards = episode.get_reward()
                
    #             # actual_batch_size = temp_batch_size * self.test_rollouts
    #             # if len(rewards) != actual_batch_size:
    #             #     logger.warning(f"Reward length mismatch: expected {actual_batch_size}, got {len(rewards)}")
    #             #     continue
                
    #             # reward_reshape = rewards.reshape((temp_batch_size, self.test_rollouts))
    #             # reward_reshape = np.sum(reward_reshape, axis=1)
    #         # ✅ temp_batch_size 是扩展后的大小（包含 rollouts）
    #             temp_batch_size = query_relation.size(0)
                
    #             # ✅ 计算原始的 batch size（不含 rollouts）
    #             original_batch_size = temp_batch_size // self.test_rollouts
                
    #             # Beam Search or Standard Rollout
    #             if beam:
    #                 self.beam_search(episode, query_relation, query_time, 
    #                                 beam_size, print_paths)
    #             else:
    #                 self.standard_rollout(episode, query_relation, query_time)
                
    #             # Calculate Rewards and Metrics
    #             rewards = episode.get_reward()
                
    #             # ✅ 修正：reward 长度应该等于 temp_batch_size
    #             if len(rewards) != temp_batch_size:
    #                 logger.warning(f"Reward length mismatch: expected {temp_batch_size}, got {len(rewards)}")
    #                 continue
                
    #             # ✅ reshape 使用原始 batch size
    #             reward_reshape = rewards.reshape((original_batch_size, self.test_rollouts))
    #             reward_reshape = np.sum(reward_reshape, axis=1)                
    #             reward_1 = (reward_reshape == self.positive_reward)
    #             reward_3 = (reward_reshape >= self.positive_reward * 0.33)
    #             reward_10 = (reward_reshape >= self.positive_reward * 0.1)
    #             reward_20 = (reward_reshape >= self.positive_reward * 0.05)
                
    #             all_final_reward_1 += np.sum(reward_1)
    #             all_final_reward_3 += np.sum(reward_3)
    #             all_final_reward_10 += np.sum(reward_10)
    #             all_final_reward_20 += np.sum(reward_20)
                
    #             # Calculate MRR
    #             for i in range(temp_batch_size):
    #                 if reward_reshape[i] > 0:
    #                     auc += 1.0
        
    #     if print_paths:
    #         path_file.close()
        
    #     self.agent.train()
    #     if self.use_gat:
    #         self.gat_model.train()
        
    #     # Normalize metrics
    #     all_final_reward_1 = all_final_reward_1 / total_examples
    #     all_final_reward_3 = all_final_reward_3 / total_examples
    #     all_final_reward_10 = all_final_reward_10 / total_examples
    #     all_final_reward_20 = all_final_reward_20 / total_examples
    #     auc = auc / total_examples
        
    #     # Log results
    #     logger.info("=" * 50)
    #     logger.info("Test Results:")
    #     logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
    #     logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
    #     logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
    #     logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
    #     logger.info("MRR: {0:7.4f}".format(auc))
    #     logger.info("=" * 50)
        
    #     # Save to file
    #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
    #         score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
    #         score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
    #         score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
    #         score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
    #         score_file.write("MRR: {0:7.4f}\n".format(auc))
    #         score_file.write("\n")
        
    #     return all_final_reward_10
    # def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
    #     """Test the model with beam search"""
    #     self.agent.eval()
    #     if self.use_gat:
    #         self.gat_model.eval()
        
    #     all_final_reward_1 = 0
    #     all_final_reward_3 = 0
    #     all_final_reward_10 = 0
    #     all_final_reward_20 = 0
    #     auc = 0
        
    #     total_examples = self.test_environment.total_no_examples
        
    #     logger.info("Starting testing on " + str(total_examples) + " examples")
        
    #     episode_num = 0
        
    #     if print_paths:
    #         path_file = codecs.open(self.path_logger_file_ + '.txt', 'a', 'utf-8')
        
    #     with torch.no_grad():
    #         for episode in tqdm(self.test_environment.get_episodes()):
    #             episode_num += 1
                
    #             query_relation = torch.tensor(
    #                 episode.get_query_relation(), 
    #                 dtype=torch.long, 
    #                 device=self.device
    #             )
    #             query_time = torch.tensor(
    #                 episode.get_query_time(), 
    #                 dtype=torch.long, 
    #                 device=self.device
    #             )
                
    #             # temp_batch_size 是扩展后的大小（包含 rollouts）
    #             temp_batch_size = query_relation.size(0)
                
    #             # 计算原始的 batch size（不含 rollouts）
    #             original_batch_size = temp_batch_size // self.test_rollouts
                
    #             # Beam Search or Standard Rollout
    #             if beam:
    #                 self.beam_search(episode, query_relation, query_time, 
    #                                 beam_size, print_paths)
    #             else:
    #                 self.standard_rollout(episode, query_relation, query_time)
                
    #             # Calculate Rewards and Metrics
    #             rewards = episode.get_reward()
                
    #             # 检查 reward 长度
    #             if len(rewards) != temp_batch_size:
    #                 logger.warning(f"Reward length mismatch: expected {temp_batch_size}, got {len(rewards)}")
    #                 continue
                
    #             # Reshape rewards: [original_batch_size, test_rollouts]
    #             reward_reshape = rewards.reshape((original_batch_size, self.test_rollouts))
    #             reward_reshape = np.sum(reward_reshape, axis=1)
                
    #             # Calculate hits at different thresholds
    #             reward_1 = (reward_reshape == self.positive_reward)
    #             reward_3 = (reward_reshape >= self.positive_reward * 0.33)
    #             reward_10 = (reward_reshape >= self.positive_reward * 0.1)
    #             reward_20 = (reward_reshape >= self.positive_reward * 0.05)
                
    #             all_final_reward_1 += np.sum(reward_1)
    #             all_final_reward_3 += np.sum(reward_3)
    #             all_final_reward_10 += np.sum(reward_10)
    #             all_final_reward_20 += np.sum(reward_20)
                
    #             # ✅ 修复：使用当前 episode 的 original_batch_size
    #             # 而不是使用之前累积的值
    #             for i in range(len(reward_reshape)):  # 或者使用 original_batch_size
    #                 if reward_reshape[i] > 0:
    #                     auc += 1.0
        
    #     if print_paths:
    #         path_file.close()
        
    #     self.agent.train()
    #     if self.use_gat:
    #         self.gat_model.train()
        
    #     # Normalize metrics
    #     all_final_reward_1 = all_final_reward_1 / total_examples
    #     all_final_reward_3 = all_final_reward_3 / total_examples
    #     all_final_reward_10 = all_final_reward_10 / total_examples
    #     all_final_reward_20 = all_final_reward_20 / total_examples
    #     auc = auc / total_examples
        
    #     # Log results
    #     logger.info("=" * 50)
    #     logger.info("Test Results:")
    #     logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
    #     logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
    #     logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
    #     logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
    #     logger.info("MRR: {0:7.4f}".format(auc))
    #     logger.info("=" * 50)
        
    #     # Save to file
    #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
    #         score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
    #         score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
    #         score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
    #         score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
    #         score_file.write("MRR: {0:7.4f}\n".format(auc))
    #         score_file.write("\n")
        
    #     return all_final_reward_10
    def test(self, beam=False, print_paths=False, save_model=True, beam_size=100):
        """Test the model"""
        self.agent.eval()
        
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0
        
        total_examples = self.test_environment.total_no_examples
        
        logger.info("Starting testing on " + str(total_examples) + " examples")
        
        episode_num = 0
        
        with torch.no_grad():
            for episode in tqdm(self.test_environment.get_episodes()):
                episode_num += 1
                
                # Prepare query
                query_relation = torch.tensor(
                    episode.get_query_relation(), 
                    dtype=torch.long, 
                    device=self.device
                )
                query_time = torch.tensor(
                    episode.get_query_time(), 
                    dtype=torch.long, 
                    device=self.device
                )
                
                temp_batch_size = query_relation.size(0)
                
                if beam:
                    # Collect trajectory using agent's policy
                    candidate_relations_list = []
                    candidate_entities_list = []
                    candidate_times_list = []
                    current_entity_seq_list = []
                    
                    state = episode.get_state()
                    range_arr = torch.arange(temp_batch_size // self.test_rollouts, device=self.device)
                    
                    for t in range(self.path_length):
                        # Collect current state
                        candidate_relations = torch.tensor(
                            state['next_relations'], dtype=torch.long, device=self.device
                        )
                        candidate_entities = torch.tensor(
                            state['next_entities'], dtype=torch.long, device=self.device
                        )
                        candidate_times = torch.tensor(
                            state['next_times'], dtype=torch.long, device=self.device
                        )
                        current_entities = torch.tensor(
                            episode.current_entities,
                            dtype=torch.long, 
                            device=self.device
                        )
                        
                        candidate_relations_list.append(candidate_relations)
                        candidate_entities_list.append(candidate_entities)
                        candidate_times_list.append(candidate_times)
                        current_entity_seq_list.append(current_entities)
                        
                        # Agent selects action
                        _, step_logits, step_action = self.agent(
                            [candidate_relations],
                            [candidate_entities],
                            [candidate_times],
                            [current_entities],
                            [None],
                            query_relation,
                            query_time,
                            range_arr,
                            first_step_of_test=True,
                            T=1
                        )
                        
                        # Extract action
                        if isinstance(step_action, list):
                            action_idx = step_action[0]
                        else:
                            action_idx = step_action
                        
                        # Take action in environment
                        state = episode(action_idx.cpu().numpy())
                    
                    # ✅ 完全重写 metrics 计算逻辑
                    rewards = episode.get_reward()
                    final_entities = episode.current_entities
                    target_entities = episode.end_entities
                    
                    # Reshape
                    num_examples = len(rewards) // self.test_rollouts
                    reward_reshape = rewards.reshape((num_examples, self.test_rollouts))
                    entities_reshape = final_entities.reshape((num_examples, self.test_rollouts))
                    targets_reshape = target_entities.reshape((num_examples, self.test_rollouts))
                    
                    # 对每个测试样本计算排名
                    from collections import defaultdict
                    
                    # for i in range(num_examples):
                    #     sample_entities = entities_reshape[i]  # [test_rollouts]
                    #     sample_rewards = reward_reshape[i]     # [test_rollouts]
                    #     target_entity = targets_reshape[i][0]  # 目标实体
                        
                    #     # ✅ 方法1：基于到达次数的加权投票
                    #     entity_scores = defaultdict(float)
                        
                    #     for j, entity_id in enumerate(sample_entities):
                    #         reward = sample_rewards[j]
                            
                    #         # 只有成功的路径才计分
                    #         if reward > 0:
                    #             entity_scores[entity_id] += 1.0
                        
                    #     # ✅ 如果没有任何成功路径，使用频率统计
                    #     if len(entity_scores) == 0:
                    #         from collections import Counter
                    #         entity_counts = Counter(sample_entities)
                    #         entity_scores = dict(entity_counts)
                        
                    #     # 按得分排序
                    #     sorted_entities = sorted(
                    #         entity_scores.items(), 
                    #         key=lambda x: x[1], 
                    #         reverse=True
                    #     )
                    #     ranked_entity_ids = [entity_id for entity_id, score in sorted_entities]
                        
                    #     # 找到目标实体的排名
                    #     if target_entity in ranked_entity_ids:
                    #         rank = ranked_entity_ids.index(target_entity) + 1
                    #     else:
                    #         # 如果目标不在成功路径中，给一个很大的排名
                    #         rank = len(self.entity_vocab) + 1
                        
                    #     # 累加指标
                    #     if rank <= 1:
                    #         all_final_reward_1 += 1
                    #     if rank <= 3:
                    #         all_final_reward_3 += 1
                    #     if rank <= 10:
                    #         all_final_reward_10 += 1
                    #     if rank <= 20:
                    #         all_final_reward_20 += 1
                        
                    #     # MRR
                    #     auc += 1.0 / rank
                    # for i in range(num_examples):
                    #     sample_entities = entities_reshape[i]  # [test_rollouts]
                    #     sample_rewards = reward_reshape[i]     # [test_rollouts]
                    #     target_entity = targets_reshape[i][0]
                        
                    #     # 🔍 调试信息
                    #     if i < 3:  # 只打印前3个例子
                    #         logger.info(f"\n{'='*50}")
                    #         logger.info(f"Example {i}:")
                    #         logger.info(f"  Target entity: {target_entity}")
                    #         logger.info(f"  Sample entities: {sample_entities[:10]}")  # 前10个
                    #         logger.info(f"  Sample rewards: {sample_rewards[:10]}")
                    #         logger.info(f"  Num successful rollouts: {np.sum(sample_rewards > 0)}")
                        
                    #     # 基于到达次数的加权投票
                    #     entity_scores = defaultdict(float)
                        
                    #     for j, entity_id in enumerate(sample_entities):
                    #         reward = sample_rewards[j]
                            
                    #         # 只有成功的路径才计分
                    #         if reward > 0:
                    #             entity_scores[entity_id] += 1.0
                        
                    #     # 🔍 调试信息
                    #     if i < 3:
                    #         logger.info(f"  Entity scores: {dict(entity_scores)}")
                    #         logger.info(f"  Num unique entities in scores: {len(entity_scores)}")
                        
                    #     # 如果没有任何成功路径，使用频率统计
                    #     if len(entity_scores) == 0:
                    #         from collections import Counter
                    #         entity_counts = Counter(sample_entities)
                    #         entity_scores = dict(entity_counts)
                        
                    #     # 按得分排序
                    #     sorted_entities = sorted(
                    #         entity_scores.items(), 
                    #         key=lambda x: x[1], 
                    #         reverse=True
                    #     )
                    #     ranked_entity_ids = [entity_id for entity_id, score in sorted_entities]
                        
                    #     # 🔍 调试信息
                    #     if i < 3:
                    #         logger.info(f"  Top 10 ranked entities: {ranked_entity_ids[:10]}")
                    #         logger.info(f"  Total ranked entities: {len(ranked_entity_ids)}")
                        
                    #     # 找到目标实体的排名
                    #     if target_entity in ranked_entity_ids:
                    #         rank = ranked_entity_ids.index(target_entity) + 1
                    #     else:
                    #         rank = len(self.entity_vocab) + 1
                        
                    #     # 🔍 调试信息
                    #     if i < 3:
                    #         logger.info(f"  Final rank: {rank}")
                    #         logger.info(f"{'='*50}\n")
                        
                    #     # 累加指标
                    #     if rank <= 1:
                    #         all_final_reward_1 += 1
                    #     if rank <= 3:
                    #         all_final_reward_3 += 1
                    #     if rank <= 10:
                    #         all_final_reward_10 += 1
                    #     if rank <= 20:
                    #         all_final_reward_20 += 1
                        
                    #     # MRR
                    #     auc += 1.0 / rank     
                    for i in range(num_examples):
                        sample_entities = entities_reshape[i]  # [test_rollouts]
                        sample_rewards = reward_reshape[i]     # [test_rollouts]
                        target_entity = targets_reshape[i][0]
                        
                        # ✅ 修复：使用所有rollout，但成功的权重更高
                        from collections import defaultdict
                        entity_scores = defaultdict(float)
                        
                        for j, entity_id in enumerate(sample_entities):
                            reward = sample_rewards[j]
                            
                            # 成功路径权重10，失败路径权重1
                            if reward > 0:
                                entity_scores[entity_id] += 10.0
                            else:
                                entity_scores[entity_id] += 1.0
                        
                        # 按得分排序
                        sorted_entities = sorted(
                            entity_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        ranked_entity_ids = [entity_id for entity_id, score in sorted_entities]
                        
                        # 🔍 调试：打印前3个例子
                        if i < 3:
                            logger.info(f"\n{'='*50}")
                            logger.info(f"Example {i}:")
                            logger.info(f"  Target: {target_entity}")
                            logger.info(f"  Entity scores (top 10): {dict(sorted_entities[:10])}")
                            logger.info(f"  Ranked entities (top 10): {ranked_entity_ids[:10]}")
                        
                        # 找到排名
                        if target_entity in ranked_entity_ids:
                            rank = ranked_entity_ids.index(target_entity) + 1
                        else:
                            rank = len(ranked_entity_ids) + 1
                        
                        # 🔍 调试
                        if i < 3:
                            logger.info(f"  Final rank: {rank}")
                            logger.info(f"{'='*50}\n")
                        
                        # 累加指标
                        if rank <= 1:
                            all_final_reward_1 += 1
                        if rank <= 3:
                            all_final_reward_3 += 1
                        if rank <= 10:
                            all_final_reward_10 += 1
                        if rank <= 20:
                            all_final_reward_20 += 1
                        
                        auc += 1.0 / rank
   
        self.agent.train()
        
        # Normalize metrics
        all_final_reward_1 = all_final_reward_1 / total_examples
        all_final_reward_3 = all_final_reward_3 / total_examples
        all_final_reward_10 = all_final_reward_10 / total_examples
        all_final_reward_20 = all_final_reward_20 / total_examples
        auc = auc / total_examples
        
        # Log results
        logger.info("=" * 50)
        logger.info("Test Results:")
        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("MRR: {0:7.4f}".format(auc))
        logger.info("=" * 50)
        
        # Save to file
        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
            score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
            score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
            score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
            score_file.write("MRR: {0:7.4f}\n".format(auc))
            score_file.write("\n")
        
        return all_final_reward_10

    # def beam_search(self, episode, query_relation, query_time, beam_size, print_paths):
    #     """Perform beam search for testing"""
    #     state = episode.get_state()
        
    #     # 获取实际的 batch size（episode 已经包含了 rollouts）
    #     actual_batch_size = len(episode.current_entities)
    #     range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
    #     # 扩展 query 到实际 batch size
    #     query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
    #     query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
        
    #     candidate_relations = []
    #     candidate_entities = []
    #     candidate_times = []
    #     current_entity_seq = []
        
    #     for t in range(self.path_length):
    #         candidate_relations.append(
    #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_entities.append(
    #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_times.append(
    #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
    #         )
    #         current_entity_seq.append(
    #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
    #         )
            
    #         # Agent forward to get action scores
    #         _, step_logits, _ = self.agent(
    #             [candidate_relations[-1]],
    #             [candidate_entities[-1]],
    #             [candidate_times[-1]],
    #             [current_entity_seq[-1]],
    #             [None],
    #             query_relation_expanded,
    #             query_time_expanded,
    #             range_arr_actual,
    #             first_step_of_test=(t == 0),
    #             T=1
    #         )
            
    #         step_log_probs = step_logits[0]
    #         action_idx = torch.argmax(step_log_probs, dim=1)
    #         state = episode(action_idx.cpu().numpy())
    # def beam_search(self, episode, query_relation, query_time, beam_size, print_paths):
    #     """Perform beam search for testing"""
    #     state = episode.get_state()
    #         # 🔍 调试信息
    #     logger.info(f"DEBUG beam_search:")
    #     logger.info(f"  query_relation.size(0): {query_relation.size(0)}")
    #     logger.info(f"  len(episode.current_entities): {len(episode.current_entities)}")
    #     logger.info(f"  test_rollouts: {self.test_rollouts}")
        
    #     # 获取实际的 batch size
    #     actual_batch_size = len(episode.current_entities)
        
    #     # ✅ 关键修复：检查 query_relation 是否已经被扩展
    #     if query_relation.size(0) != actual_batch_size:
    #         # 需要扩展
    #         query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
    #         query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
    #     else:
    #         # 已经是正确的大小
    #         query_relation_expanded = query_relation
    #         query_time_expanded = query_time
        
    #     range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
    #     candidate_relations = []
    #     candidate_entities = []
    #     candidate_times = []
    #     current_entity_seq = []
        
    #     for t in range(self.path_length):
    #         candidate_relations.append(
    #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_entities.append(
    #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_times.append(
    #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
    #         )
    #         current_entity_seq.append(
    #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
    #         )
            
    #         # ✅ 确保所有维度匹配
    #         assert candidate_relations[-1].size(0) == actual_batch_size, \
    #             f"candidate_relations size {candidate_relations[-1].size(0)} != {actual_batch_size}"
    #         assert current_entity_seq[-1].size(0) == actual_batch_size, \
    #             f"current_entity_seq size {current_entity_seq[-1].size(0)} != {actual_batch_size}"
    #         assert query_relation_expanded.size(0) == actual_batch_size, \
    #             f"query_relation_expanded size {query_relation_expanded.size(0)} != {actual_batch_size}"
            
    #         # Agent forward
    #         _, step_logits, _ = self.agent(
    #             [candidate_relations[-1]],
    #             [candidate_entities[-1]],
    #             [candidate_times[-1]],
    #             [current_entity_seq[-1]],
    #             [None],
    #             query_relation_expanded,
    #             query_time_expanded,
    #             range_arr_actual,
    #             first_step_of_test=(t == 0),
    #             T=1
    #         )
            
    #         step_log_probs = step_logits[0]
    #         action_idx = torch.argmax(step_log_probs, dim=1)
    #         state = episode(action_idx.cpu().numpy())

    # def standard_rollout(self, episode, query_relation, query_time):
    #     """Standard rollout for testing (without beam search)"""
    #     state = episode.get_state()
        
    #     actual_batch_size = len(episode.current_entities)
    #     range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
    #     query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
    #     query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
        
    #     candidate_relations = []
    #     candidate_entities = []
    #     candidate_times = []
    #     current_entity_seq = []
        
    #     for t in range(self.path_length):
    #         candidate_relations.append(
    #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_entities.append(
    #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_times.append(
    #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
    #         )
    #         current_entity_seq.append(
    #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
    #         )
            
    #         _, _, step_action = self.agent(
    #             [candidate_relations[-1]],
    #             [candidate_entities[-1]],
    #             [candidate_times[-1]],
    #             [current_entity_seq[-1]],
    #             [None],
    #             query_relation_expanded,
    #             query_time_expanded,
    #             range_arr_actual,
    #             first_step_of_test=(t == 0),
    #             T=1
    #         )
            
    #         if isinstance(step_action, list):
    #             action_idx = step_action[0]
    #         else:
    #             action_idx = step_action
            
    #         state = episode(action_idx.cpu().numpy())
    # def standard_rollout(self, episode, query_relation, query_time):
    #     """Standard rollout for testing"""
    #     state = episode.get_state()
        
    #     actual_batch_size = len(episode.current_entities)
        
    #     # ✅ 同样的检查和扩展逻辑
    #     if query_relation.size(0) != actual_batch_size:
    #         query_relation_expanded = query_relation.repeat_interleave(self.test_rollouts)
    #         query_time_expanded = query_time.repeat_interleave(self.test_rollouts)
    #     else:
    #         query_relation_expanded = query_relation
    #         query_time_expanded = query_time
        
    #     range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
    #     candidate_relations = []
    #     candidate_entities = []
    #     candidate_times = []
    #     current_entity_seq = []
        
    #     for t in range(self.path_length):
    #         candidate_relations.append(
    #             torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_entities.append(
    #             torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
    #         )
    #         candidate_times.append(
    #             torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
    #         )
    #         current_entity_seq.append(
    #             torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
    #         )
            
    #         _, _, step_action = self.agent(
    #             [candidate_relations[-1]],
    #             [candidate_entities[-1]],
    #             [candidate_times[-1]],
    #             [current_entity_seq[-1]],
    #             [None],
    #             query_relation_expanded,
    #             query_time_expanded,
    #             range_arr_actual,
    #             first_step_of_test=(t == 0),
    #             T=1
    #         )
            
    #         if isinstance(step_action, list):
    #             action_idx = step_action[0]
    #         else:
    #             action_idx = step_action
            
    #         state = episode(action_idx.cpu().numpy())
    def beam_search(self, episode, query_relation, query_time, beam_size, print_paths):
        """Perform beam search for testing"""
        state = episode.get_state()
        
        # 获取实际的 batch size
        actual_batch_size = len(episode.current_entities)
        
        # ✅ 关键：query_relation 已经是正确大小，不需要扩展
        query_relation_expanded = query_relation
        query_time_expanded = query_time
        
        range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
        # 🔍 调试信息（可以后续删除）
        logger.info(f"DEBUG beam_search:")
        logger.info(f"  query_relation.size(0): {query_relation.size(0)}")
        logger.info(f"  len(episode.current_entities): {len(episode.current_entities)}")
        logger.info(f"  actual_batch_size: {actual_batch_size}")
        
        candidate_relations = []
        candidate_entities = []
        candidate_times = []
        current_entity_seq = []
        
        for t in range(self.path_length):
            candidate_relations.append(
                torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
            )
            candidate_entities.append(
                torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
            )
            candidate_times.append(
                torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
            )
            current_entity_seq.append(
                torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
            )
            
            # Agent forward
            _, step_logits, _ = self.agent(
                [candidate_relations[-1]],
                [candidate_entities[-1]],
                [candidate_times[-1]],
                [current_entity_seq[-1]],
                [None],
                query_relation_expanded,
                query_time_expanded,
                range_arr_actual,
                first_step_of_test=(t == 0),
                T=1
            )
            
            step_log_probs = step_logits[0]
            action_idx = torch.argmax(step_log_probs, dim=1)
            state = episode(action_idx.cpu().numpy())
    def standard_rollout(self, episode, query_relation, query_time):
        """Standard rollout for testing"""
        state = episode.get_state()
        
        actual_batch_size = len(episode.current_entities)
        
        # ✅ 不需要扩展
        query_relation_expanded = query_relation
        query_time_expanded = query_time
        
        range_arr_actual = torch.arange(actual_batch_size, device=self.device)
        
        candidate_relations = []
        candidate_entities = []
        candidate_times = []
        current_entity_seq = []
        
        for t in range(self.path_length):
            candidate_relations.append(
                torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
            )
            candidate_entities.append(
                torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
            )
            candidate_times.append(
                torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
            )
            current_entity_seq.append(
                torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
            )
            
            _, _, step_action = self.agent(
                [candidate_relations[-1]],
                [candidate_entities[-1]],
                [candidate_times[-1]],
                [current_entity_seq[-1]],
                [None],
                query_relation_expanded,
                query_time_expanded,
                range_arr_actual,
                first_step_of_test=(t == 0),
                T=1
            )
            
            if isinstance(step_action, list):
                action_idx = step_action[0]
            else:
                action_idx = step_action
            
            state = episode(action_idx.cpu().numpy())
    
    def save_model(self, path):
        """Save model checkpoint"""
        save_dict = {
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_agent_state_dict': self.optimizer_agent.state_dict(),
            'batch_counter': self.batch_counter,
            'baseline': self.baseline.b,
        }
        
        if self.use_gat:
            save_dict['gat_state_dict'] = self.gat_model.state_dict()
            save_dict['optimizer_gat_state_dict'] = self.optimizer_gat.state_dict()
        
        torch.save(save_dict, path)
        logger.info("Model saved to: " + path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer_agent.load_state_dict(checkpoint['optimizer_agent_state_dict'])
        
        if self.use_gat and 'gat_state_dict' in checkpoint:
            self.gat_model.load_state_dict(checkpoint['gat_state_dict'])
            self.optimizer_gat.load_state_dict(checkpoint['optimizer_gat_state_dict'])
        
        self.batch_counter = checkpoint['batch_counter']
        if 'baseline' in checkpoint:
            self.baseline.b = checkpoint['baseline']
        
        logger.info("Model loaded from: " + path)


if __name__ == '__main__':
    options = read_options()
    
    # Setup logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    if not os.path.exists(options['base_output_dir']):
        os.makedirs(options['base_output_dir'])
    
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    
    # Load vocabs
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    options['time_vocab'] = json.load(open(options['vocab_dir'] + '/time_vocab.json'))
    
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    logger.info('Total number of times {}'.format(len(options['time_vocab'])))
    
    # Create trainer with GAT
    trainer = Trainer(options)
    
    if not options['load_model']:
        trainer.initialize_pretrained_embeddings()
        trainer.train()
    else:
        trainer.load_model(options['model_load_dir'])
        trainer.test(beam=True, print_paths=True, save_model=False)
