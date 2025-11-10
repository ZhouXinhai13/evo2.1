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

from src.model.agent import Agent  # PyTorch版本
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
        
        # 创建模型
        self.agent = Agent(params).to(self.device)
        
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
        
        # # Optimizer
        # self.optimizer = optim.Adam(
        #     self.agent.parameters(), 
        #     lr=self.learning_rate,
        #     weight_decay=self.l2_reg_const
        # )
        # ===== 🔥 优化器改进：为时间编码器设置更高的学习率 =====
        print("\n" + "="*70)
        print("🔧 配置优化器（差异化学习率）")
        print("="*70)

        time_encoder_params = []
        other_params = []

        # 分离时间编码器和其他参数
        for name, param in self.agent.named_parameters():
            if 'time_diff_encoder' in name:
                time_encoder_params.append(param)
                logger.info(f"  🔥 时间编码器参数: {name} (shape: {param.shape})")
            else:
                other_params.append(param)

        # 统计参数数量
        num_time_params = sum(p.numel() for p in time_encoder_params)
        num_other_params = sum(p.numel() for p in other_params)
        total_params = num_time_params + num_other_params

        logger.info(f"\n  📊 参数统计:")
        logger.info(f"     时间编码器参数: {num_time_params:,} ({num_time_params/total_params*100:.1f}%)")
        logger.info(f"     其他参数: {num_other_params:,} ({num_other_params/total_params*100:.1f}%)")
        logger.info(f"     总参数: {total_params:,}")

        # 创建优化器（差异化学习率）
        time_lr_multiplier = 1.0  # 时间编码器学习率倍数

        self.optimizer = optim.Adam([
            {
                'params': other_params, 
                'lr': self.learning_rate,
                'weight_decay': self.l2_reg_const
            },
            {
                'params': time_encoder_params, 
                'lr': self.learning_rate * time_lr_multiplier,
                'weight_decay': self.l2_reg_const * 0.1
            }
        ])

        logger.info(f"\n  ✅ 优化器配置完成:")
        logger.info(f"     基础学习率: {self.learning_rate:.6f}")
        logger.info(f"     时间编码器学习率: {self.learning_rate * time_lr_multiplier:.6f} ({time_lr_multiplier}x)")
        logger.info(f"     权重衰减: {self.l2_reg_const}")
        print("="*70 + "\n")        
        # Counters
        self.batch_counter = 0
        self.global_step = 0
        
        # Path logger
        self.path_logger_file_ = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def get_decaying_beta(self):
        """Exponential decay of beta"""
        return self.beta * (0.90 ** (self.global_step / 200))
    
    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        """
        Calculate REINFORCE loss with baseline
        
        Args:
            per_example_loss: list of [batch_size] tensors
            per_example_logits: list of [batch_size, max_actions] tensors
            cum_discounted_reward: [batch_size, path_length] tensor
        """
        # Stack losses: [batch_size, path_length]
        loss = torch.stack(per_example_loss, dim=1)
        
        # Get baseline
        baseline_value = self.baseline.get_baseline_value()
        
        # Calculate advantage
        final_reward = cum_discounted_reward - baseline_value
        
        # Normalize reward per batch
        reward_mean = final_reward.mean()
        reward_std = final_reward.std() + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std
        
        # Apply advantage to loss
        loss = loss * final_reward
        
        # Calculate entropy regularization
        entropy_loss = self.entropy_reg_loss(per_example_logits)
        
        # Total loss
        decaying_beta = self.get_decaying_beta()
        total_loss = loss.mean() - decaying_beta * entropy_loss
        
        return total_loss
    # def check_time_embedding_gradient(self):
        """检查时间嵌入的梯度信息（详细版）"""
        time_emb_layer = self.agent.time_lookup_table
        
        # 1️⃣ 检查梯度是否存在
        if time_emb_layer.weight.grad is None:
            logger.warning("⚠️  时间嵌入的梯度为 None！")
            logger.warning(f"    requires_grad 状态: {time_emb_layer.weight.requires_grad}")
            return
        
        # 2️⃣ 计算梯度统计量
        grad = time_emb_layer.weight.grad
        grad_norm = grad.norm().item()              # L2范数
        grad_mean = grad.mean().item()              # 平均值
        grad_std = grad.std().item()                # 标准差
        grad_max = grad.abs().max().item()          # 最大绝对值
        grad_min = grad.abs().min().item()          # 最小绝对值
        
        # 3️⃣ 统计非零梯度的比例
        num_nonzero = (grad.abs() > 1e-8).sum().item()
        total_params = grad.numel()
        nonzero_ratio = num_nonzero / total_params * 100
        
        # 4️⃣ 输出详细信息
        logger.info("="*70)
        logger.info(f"Batch {self.batch_counter} - 时间嵌入梯度分析")
        logger.info("="*70)
        logger.info(f"  梯度范数 (Grad Norm):       {grad_norm:.6f}")
        logger.info(f"  梯度均值 (Grad Mean):       {grad_mean:.9f}")
        logger.info(f"  梯度标准差 (Grad Std):      {grad_std:.6f}")
        logger.info(f"  梯度最大值 (Grad Max):      {grad_max:.6f}")
        logger.info(f"  梯度最小值 (Grad Min):      {grad_min:.9f}")
        logger.info(f"  非零梯度比例:               {nonzero_ratio:.2f}%")
        logger.info(f"  参数总数:                   {total_params:,}")
        logger.info(f"  非零梯度数:                 {num_nonzero:,}")
        
        # 5️⃣ 健康度诊断
        self.diagnose_gradient_health(grad_norm, grad_mean, grad_std, nonzero_ratio)
        
        logger.info("="*70 + "\n")
    def check_time_embedding_gradient(self):
        """
        检查时间编码层的梯度（支持新旧两种架构）
        """
        logger.info("="*70)
        logger.info(f"Batch {self.batch_counter} - 时间编码梯度分析")
        logger.info("="*70)
        
        # ===== 🔥 适配新架构：time_diff_encoder =====
        if hasattr(self.agent, 'time_diff_encoder'):
            logger.info("  ✅ 使用相对时间编码器 (time_diff_encoder)")
            self._check_time_diff_encoder_gradients()
        
        # 兼容旧架构：time_lookup_table
        elif hasattr(self.agent, 'time_lookup_table'):
            logger.info("  ✅ 使用时间查找表 (time_lookup_table)")
            self._check_time_lookup_table_gradients()
        
        else:
            logger.error("  ❌ Agent 没有任何时间编码层！")
        
        logger.info("="*70 + "\n")

    def _check_time_diff_encoder_gradients(self):
        """检查 time_diff_encoder (MLP) 的梯度"""
        has_any_grad = False
        
        for idx, layer in enumerate(self.agent.time_diff_encoder):
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    has_any_grad = True
                    
                    grad = layer.weight.grad
                    grad_norm = grad.norm().item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    grad_max = grad.abs().max().item()
                    
                    num_nonzero = (grad.abs() > 1e-8).sum().item()
                    total_params = grad.numel()
                    nonzero_ratio = num_nonzero / total_params * 100
                    
                    logger.info(f"\n  📊 time_diff_encoder[{idx}] (Linear层):")
                    logger.info(f"     权重形状: {layer.weight.shape}")
                    logger.info(f"     梯度范数: {grad_norm:.6f}")
                    logger.info(f"     梯度均值: {grad_mean:.9f}")
                    logger.info(f"     梯度标准差: {grad_std:.6f}")
                    logger.info(f"     梯度最大值: {grad_max:.6f}")
                    logger.info(f"     非零梯度比例: {nonzero_ratio:.2f}%")
                    
                    # 健康度检查
                    if grad_norm < 1e-6:
                        logger.warning("     ⚠️  梯度接近零！")
                    elif grad_norm > 100:
                        logger.warning("     ⚠️  梯度过大！")
                    else:
                        logger.info("     ✅ 梯度健康")
                    
                    if nonzero_ratio < 10:
                        logger.warning("     ⚠️  大部分参数未更新")
                else:
                    logger.warning(f"\n  ❌ time_diff_encoder[{idx}]: 无梯度")
        
        if not has_any_grad:
            logger.error("  ⚠️  警告：time_diff_encoder 的所有层都没有梯度！")

    def _check_time_lookup_table_gradients(self):
        """检查 time_lookup_table (Embedding) 的梯度"""
        time_emb_layer = self.agent.time_lookup_table
        
        if time_emb_layer.weight.grad is None:
            logger.warning("  ⚠️  时间嵌入的梯度为 None！")
            logger.warning(f"     requires_grad 状态: {time_emb_layer.weight.requires_grad}")
            return
        
        grad = time_emb_layer.weight.grad
        grad_norm = grad.norm().item()
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_max = grad.abs().max().item()
        grad_min = grad.abs().min().item()
        
        num_nonzero = (grad.abs() > 1e-8).sum().item()
        total_params = grad.numel()
        nonzero_ratio = num_nonzero / total_params * 100
        
        logger.info(f"\n  📊 time_lookup_table:")
        logger.info(f"     权重形状: {time_emb_layer.weight.shape}")
        logger.info(f"     梯度范数: {grad_norm:.6f}")
        logger.info(f"     梯度均值: {grad_mean:.9f}")
        logger.info(f"     梯度标准差: {grad_std:.6f}")
        logger.info(f"     梯度最大值: {grad_max:.6f}")
        logger.info(f"     梯度最小值: {grad_min:.9f}")
        logger.info(f"     非零梯度比例: {nonzero_ratio:.2f}%")
        
        # 健康度检查
        if grad_norm < 1e-6:
            logger.warning("     ⚠️  梯度接近零！")
        elif grad_norm > 100:
            logger.warning("     ⚠️  梯度过大！")
        else:
            logger.info("     ✅ 梯度健康")
        
        if nonzero_ratio < 10:
            logger.warning("     ⚠️  大部分参数未更新")


    def diagnose_gradient_health(self, grad_norm, grad_mean, grad_std, nonzero_ratio):
        """诊断梯度健康状况"""
        logger.info("\n  梯度健康度诊断:")
        
        # 检查梯度消失
        if grad_norm < 1e-6:
            logger.warning("    ⚠️  梯度接近零！可能存在梯度消失问题")
        elif grad_norm < 1e-4:
            logger.info("    ⚡ 梯度较小，但在正常范围")
        else:
            logger.info("    ✅ 梯度范数正常")
        
        # 检查梯度爆炸
        if grad_norm > 10:
            logger.warning("    ⚠️  梯度过大！可能存在梯度爆炸")
        
        # 检查非零梯度比例
        if nonzero_ratio < 10:
            logger.warning("    ⚠️  非零梯度比例过低！大部分参数未更新")
        elif nonzero_ratio > 90:
            logger.info("    ✅ 大部分参数都在更新")
        else:
            logger.info(f"    ⚡ {nonzero_ratio:.1f}% 的参数在更新")
        
        # 检查梯度分布
        if grad_std < 1e-6:
            logger.warning("    ⚠️  梯度方差过小！梯度几乎是常量")
        else:
            logger.info("    ✅ 梯度分布正常")


    # def check_all_embedding_gradients(self):
        """对比所有嵌入层的梯度"""
        logger.info("\n" + "="*70)
        logger.info(f"Batch {self.batch_counter} - 所有嵌入层梯度对比")
        logger.info("="*70)
        
        # 检查实体嵌入
        entity_grad = self.agent.entity_lookup_table.weight.grad
        if entity_grad is not None:
            entity_norm = entity_grad.norm().item()
            entity_mean = entity_grad.mean().item()
            entity_nonzero = (entity_grad.abs() > 1e-8).sum().item()
            entity_total = entity_grad.numel()
            entity_ratio = entity_nonzero / entity_total * 100
            logger.info(f"实体嵌入 (Entity):")
            logger.info(f"  norm={entity_norm:.6f}, mean={entity_mean:.9f}, nonzero={entity_ratio:.2f}%")
        else:
            logger.info(f"实体嵌入 (Entity):  梯度为 None")
        
        # 检查关系嵌入
        relation_grad = self.agent.relation_lookup_table.weight.grad
        if relation_grad is not None:
            relation_norm = relation_grad.norm().item()
            relation_mean = relation_grad.mean().item()
            relation_nonzero = (relation_grad.abs() > 1e-8).sum().item()
            relation_total = relation_grad.numel()
            relation_ratio = relation_nonzero / relation_total * 100
            logger.info(f"关系嵌入 (Relation):")
            logger.info(f"  norm={relation_norm:.6f}, mean={relation_mean:.9f}, nonzero={relation_ratio:.2f}%")
        else:
            logger.info(f"关系嵌入 (Relation): 梯度为 None")
        
        # 检查时间嵌入
        time_grad = self.agent.time_lookup_table.weight.grad
        if time_grad is not None:
            time_norm = time_grad.norm().item()
            time_mean = time_grad.mean().item()
            time_nonzero = (time_grad.abs() > 1e-8).sum().item()
            time_total = time_grad.numel()
            time_ratio = time_nonzero / time_total * 100
            logger.info(f"时间嵌入 (Time):")
            logger.info(f"  norm={time_norm:.6f}, mean={time_mean:.9f}, nonzero={time_ratio:.2f}%")
        else:
            logger.info(f"时间嵌入 (Time):     梯度为 None")
        
        logger.info("="*70 + "\n")
    def check_all_embedding_gradients(self):
        """对比所有嵌入层的梯度"""
        logger.info("\n" + "="*70)
        logger.info(f"Batch {self.batch_counter} - 所有嵌入层梯度对比")
        logger.info("="*70)
        
        # 检查实体嵌入
        self._log_embedding_grad("实体嵌入 (Entity)", self.agent.entity_lookup_table)
        
        # 检查关系嵌入
        self._log_embedding_grad("关系嵌入 (Relation)", self.agent.relation_lookup_table)
        
        # 检查时间编码（适配新架构）
        if hasattr(self.agent, 'time_diff_encoder'):
            logger.info("\n时间编码 (Time - MLP):")
            for idx, layer in enumerate(self.agent.time_diff_encoder):
                if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    grad_mean = layer.weight.grad.mean().item()
                    nonzero_ratio = ((layer.weight.grad.abs() > 1e-8).sum().item() / 
                                    layer.weight.grad.numel() * 100)
                    logger.info(f"  Layer[{idx}]: norm={grad_norm:.6f}, mean={grad_mean:.9f}, "
                            f"nonzero={nonzero_ratio:.2f}%")
        elif hasattr(self.agent, 'time_lookup_table'):
            self._log_embedding_grad("时间嵌入 (Time)", self.agent.time_lookup_table)
        
        logger.info("="*70 + "\n")

    def _log_embedding_grad(self, name, embedding_layer):
        """辅助函数：记录单个嵌入层的梯度"""
        grad = embedding_layer.weight.grad
        
        if grad is not None:
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            nonzero = (grad.abs() > 1e-8).sum().item()
            total = grad.numel()
            ratio = nonzero / total * 100
            
            logger.info(f"\n{name}:")
            logger.info(f"  norm={grad_norm:.6f}, mean={grad_mean:.9f}, nonzero={ratio:.2f}%")
        else:
            logger.info(f"\n{name}: 梯度为 None")
    def check_time_feature_usage(self):
        """
        检查模型是否真的在使用时间特征
        """
        logger.info("\n" + "="*70)
        logger.info(f"Batch {self.batch_counter} - 时间特征使用情况分析")
        logger.info("="*70)
        
        if hasattr(self.agent, 'time_diff_encoder'):
            with torch.no_grad():
                # 创建测试输入：不同的时间差
                test_time_diffs = torch.tensor([
                    [-0.5],  # 未来
                    [-0.1],  # 近未来
                    [0.0],   # 当前
                    [0.1],   # 近过去
                    [0.5]    # 远过去
                ], dtype=torch.float32, device=self.device)
                
                # 通过编码器
                time_embeddings = self.agent.time_diff_encoder(test_time_diffs)
                
                # 分析输出的多样性
                emb_std = time_embeddings.std(dim=0).mean().item()
                emb_range = (time_embeddings.max() - time_embeddings.min()).item()
                
                logger.info(f"\n  📊 时间编码器输出分析:")
                logger.info(f"     不同时间差的嵌入标准差: {emb_std:.6f}")
                logger.info(f"     嵌入值范围: {emb_range:.6f}")
                
                if emb_std < 0.01:
                    logger.warning("     ❌ 所有时间的嵌入几乎相同")
                elif emb_std < 0.1:
                    logger.info("     ⚠️  时间嵌入多样性较低")
                else:
                    logger.info("     ✅ 时间嵌入有良好的区分度")
                
                # 检查对"未来"的敏感性
                future_emb = time_embeddings[0]
                current_emb = time_embeddings[2]
                
                future_vs_current = (future_emb - current_emb).norm().item()
                logger.info(f"\n  📊 未来 vs 当前的差异:")
                logger.info(f"     L2距离: {future_vs_current:.6f}")
                
                if future_vs_current < 0.1:
                    logger.warning("     ❌ 模型无法区分未来和当前")
                else:
                    logger.info("     ✅ 模型能够区分未来和当前")
        
        logger.info("="*70 + "\n")

    def entropy_reg_loss(self, all_logits):
        """Calculate entropy regularization"""
        # Stack: [batch_size, max_actions, path_length]
        all_logits = torch.stack(all_logits, dim=2)
        
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(torch.exp(all_logits) * all_logits, dim=1)
        entropy_mean = entropy.mean()
        
        return entropy_mean
    
    def calc_cum_discounted_reward(self, rewards):
        """
        Calculate cumulative discounted reward
        
        Args:
            rewards: [batch_size] numpy array
        
        Returns:
            cum_disc_reward: [batch_size, path_length] numpy array
        """
        batch_size = len(rewards)
        running_add = np.zeros(batch_size)
        cum_disc_reward = np.zeros((batch_size, self.path_length))
        
        cum_disc_reward[:, self.path_length - 1] = rewards
        
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        
        return cum_disc_reward
    
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
    '''   
    # def train(self):
    #     """Main training loop"""
    #     train_loss = 0.0
    #     self.batch_counter = 0
        
    #     logger.info("Starting training...")
        
    #     for episode in self.train_environment.get_episodes():
    #         self.batch_counter += 1
    #         self.global_step += 1
    #         
    #         # Prepare query
    #         # query_relation = torch.tensor(
    #         #     episode.get_query_relation(), 
    #         #     dtype=torch.long, 
    #         #     device=self.device
    #         # )
    #         # query_time = torch.tensor(
    #         #     episode.get_query_time(), 
    #         #     dtype=torch.long, 
    #         #     device=self.device
    #         # )
            
    #         # # Collect trajectory
    #         # candidate_relations = []
    #         # candidate_entities = []
    #         # candidate_times = []
    #         # current_entity_seq = []
            
    #         # state = episode.get_state()
            
    #         # for t in range(self.path_length):
    #         #     candidate_relations.append(
    #         #         torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
    #         #     )
    #         #     candidate_entities.append(
    #         #         torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
    #         #     )
    #         #     candidate_times.append(
    #         #         torch.tensor(state['next_times'], dtype=torch.long, device=self.device)
    #         #     )
    #         #     current_entity_seq.append(
    #         #         torch.tensor(state['current_entities'], dtype=torch.long, device=self.device)
    #         #     )
                
    #         #     # Take a step (agent will sample action)
    #         #     # For now, use dummy action to advance episode
    #         #     # state = episode(np.zeros(self.batch_size * self.num_rollouts, dtype='int32'))
    #         #     num_actions = state['next_relations'].shape[1]
    #         #     random_actions = np.random.randint(0, num_actions, size=self.batch_size * self.num_rollouts, dtype='int32')
    #         #     state = episode(random_actions)
            
    #         # # Forward pass through agent
    #         # range_arr = torch.arange(self.batch_size, device=self.device)
            
    #         # # Path labels (not used during RL training, set to zeros)
    #         # path_labels = [None for _ in range(self.path_length)]

            
    #         # per_example_loss, per_example_logits, action_idx = self.agent(
    #         #     candidate_relations,
    #         #     candidate_entities,
    #         #     candidate_times,
    #         #     current_entity_seq,
    #         #     path_labels,
    #         #     query_relation,
    #         #     query_time,
    #         #     range_arr,
    #         #     first_step_of_test=False,
    #         #     T=self.path_length
    #         # )
    #         
    #                 # ✅ 在这里添加：Episode 开始时打印
    #         if self.batch_counter <= 3:  # 只打印前3个batch，避免刷屏
    #             logger.info(f"\n{'='*60}")
    #             logger.info(f"BATCH {self.batch_counter} - 时间信息调试")
    #             logger.info(f"{'='*60}")
                
    #             # 打印查询时间
    #             query_times = episode.get_query_time()
    #             logger.info(f"Query times (shape {query_times.shape}): {query_times[:5]}")  # 只显示前5个
                
    #             # 打印目标实体（用于对比）
    #             logger.info(f"Target entities: {episode.end_entities[:5]}")
    #         query_relation = torch.tensor(
    #             episode.get_query_relation(), 
    #             dtype=torch.long, 
    #             device=self.device
    #         )
    #         query_time = torch.tensor(
    #             episode.get_query_time(), 
    #             dtype=torch.long, 
    #             device=self.device
    #         )
            
    #         # Collect trajectory with agent making decisions at each step
    #         candidate_relations = []
    #         candidate_entities = []
    #         candidate_times = []
    #         current_entity_seq = []
    #         all_actions = []
    #         all_logits_list = []
            
    #         state = episode.get_state()
    #         range_arr = torch.arange(self.batch_size, device=self.device)
            
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
    #             # current_entity_seq.append(
    #             #     torch.tensor(state['current_entities'], dtype=torch.long, device=self.device)
    #             # )
    #             # 修改后
    #             current_entity_seq.append(
    #                 torch.tensor(episode.current_entities, dtype=torch.long, device=self.device)
    #             )
    #             # Agent selects action based on current state
    #             # We need to detach here because we'll do a separate forward pass for loss
    #             with torch.no_grad():
    #                 _, step_logits, step_action = self.agent(
    #                     [candidate_relations[-1]],  # Only current step
    #                     [candidate_entities[-1]],
    #                     [candidate_times[-1]],
    #                     [current_entity_seq[-1]],
    #                     [None],  # No label for RL
    #                     query_relation,
    #                     query_time,
    #                     range_arr,
    #                     first_step_of_test=False,
    #                     T=1  # Single step
    #                 )
                    
    #                 # Extract action for this step
    #                 if isinstance(step_action, list):
    #                     action_idx = step_action[0]
    #                 else:
    #                     action_idx = step_action
                    
    #                 all_actions.append(action_idx)
                
    #             # Take action in environment
    #             state = episode(action_idx.cpu().numpy())
            
    #         # Get final rewards
    #         rewards = episode.get_reward()
            
    #         # Now do a full forward pass for computing loss
    #         path_labels = [None for _ in range(self.path_length)]
    #         per_example_loss, per_example_logits, _ = self.agent(
    #             candidate_relations,
    #             candidate_entities,
    #             candidate_times,
    #             current_entity_seq,
    #             path_labels,
    #             query_relation,
    #             query_time,
    #             range_arr,
    #             first_step_of_test=False,
    #             T=self.path_length
    #         )

    #         # Get rewards from environment
    #         # rewards = episode.get_reward()
            
    #         # Calculate cumulative discounted reward
    #         cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
    #         cum_discounted_reward = torch.tensor(
    #             cum_discounted_reward, 
    #             dtype=torch.float32, 
    #             device=self.device
    #         )
            
    #         # Calculate loss
    #         loss = self.calc_reinforce_loss(
    #             per_example_loss, 
    #             per_example_logits, 
    #             cum_discounted_reward
    #         )
            
    #         # Backward pass
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
    #         # ✅ 调用详细版本（每10个batch检查时间嵌入）
    #         if self.batch_counter % 10 == 0:
    #             self.check_time_embedding_gradient()

    #         # ✅ 调用对比版本（每50个batch对比所有嵌入）
    #         if self.batch_counter % 50 == 0:
    #             self.check_all_embedding_gradients()
    #             self.check_time_feature_usage()

    #         self.optimizer.step()
            
    #         # Update baseline
    #         self.baseline.update(cum_discounted_reward.mean().item())
''' 
    def train(self):
        """Main training loop - 修正版"""
        train_loss = 0.0
        self.batch_counter = 0
        
        logger.info("Starting training...")
        
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
            
            # ===== 🔥 关键改动: 单次前向传播,保留梯度 =====
            candidate_relations = []
            candidate_entities = []
            candidate_times = []
            current_entity_seq = []
            
            # 用于存储每一步的 log_prob (REINFORCE核心)
            log_probs = []
            
            state = episode.get_state()
            range_arr = torch.arange(self.batch_size, device=self.device)
            
            # ===== Rollout: 不使用 no_grad() =====
            for t in range(self.path_length):
                # 收集当前状态
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
                
                # 🔥 关键: 不使用 no_grad(),让梯度可以流动!
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
                
                # 提取 action_idx
                if isinstance(step_action, list):
                    action_idx = step_action[0]
                else:
                    action_idx = step_action
                
                # 🔥 关键: 计算这个动作的 log_prob
                # step_logits[0] 是 [batch_size, num_actions] 的 log softmax
                action_log_prob = step_logits[0].gather(1, action_idx.unsqueeze(1)).squeeze(1)
                log_probs.append(action_log_prob)
                
                # 推进环境 (使用 .detach() 避免环境操作影响梯度)
                state = episode(action_idx.detach().cpu().numpy())
            
            # ===== 计算奖励和损失 =====
            rewards = episode.get_reward()
            
            # 计算累积折扣奖励
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            cum_discounted_reward = torch.tensor(
                cum_discounted_reward, 
                dtype=torch.float32, 
                device=self.device
            )  # [batch_size, path_length]
            
            # Baseline
            baseline_value = self.baseline.get_baseline_value()
            advantage = cum_discounted_reward - baseline_value
            
            # 归一化 advantage (per batch)
            advantage_mean = advantage.mean(dim=0, keepdim=True)
            advantage_std = advantage.std(dim=0, keepdim=True) + 1e-6
            advantage = (advantage - advantage_mean) / advantage_std
            
            # 🔥 REINFORCE 损失: -log_prob * advantage
            # log_probs: list of [batch_size], length = path_length
            # advantage: [batch_size, path_length]
            reinforce_loss = 0
            for t in range(self.path_length):
                reinforce_loss += -(log_probs[t] * advantage[:, t]).mean()
            
            # Entropy regularization (可选)
            # 这里省略,可以从 per_example_logits 中计算
            
            # 总损失
            loss = reinforce_loss
            
            # ===== 反向传播 =====
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            
            # 梯度检查 (保持您原有的)
            if self.batch_counter % 1000 == 0:
                self.check_time_embedding_gradient()
            
            if self.batch_counter % 1000 == 0:
                self.check_all_embedding_gradients()
                self.check_time_feature_usage()
            
            self.optimizer.step()
            
            # Update baseline
            self.baseline.update(cum_discounted_reward.mean().item())
            
            # ... [后续统计和评估代码保持不变] ...
            
            # Statistics
            train_loss = 0.98 * train_loss + 0.02 * loss.item()
            avg_reward = np.mean(rewards)
            
            # Calculate correct episodes
            num_episodes = len(rewards) // self.num_rollouts
            reward_reshape = np.reshape(rewards, (num_episodes, self.num_rollouts))
            reward_reshape = np.sum(reward_reshape, axis=1)
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")
            
            logger.info(
                "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
                "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".format(
                    self.batch_counter, np.sum(rewards), avg_reward,
                    num_ep_correct, (num_ep_correct / num_episodes), train_loss
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
                
                # Save model
                self.save_model(self.model_dir + '/model_' + str(self.batch_counter) + '.pt')
            
            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            
            gc.collect()
            
            if self.batch_counter >= self.total_iterations:
                break
    
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
                
                # # Beam search
                # if beam:
                #     # Implement beam search (simplified version)
                #     # Collect all logits for final ranking
                #     all_logits = []
                    
                #     for t in range(self.path_length):
                #         state = episode.get_state()
                        
                #         candidate_relations = torch.tensor(
                #             state['next_relations'], dtype=torch.long, device=self.device
                #         )
                #         candidate_entities = torch.tensor(
                #             state['next_entities'], dtype=torch.long, device=self.device
                #         )
                #         candidate_times = torch.tensor(
                #             state['next_times'], dtype=torch.long, device=self.device
                #         )
                #         current_entities = torch.tensor(
                #             state['current_entities'], dtype=torch.long, device=self.device
                #         )
                        
                #         # Get action probabilities
                #         # (Simplified - in full version would maintain beam)
                #         state = episode(np.zeros(temp_batch_size, dtype='int32'))
                    

                # Beam search
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
                        # current_entities = torch.tensor(
                        #     state['current_entities'], dtype=torch.long, device=self.device
                        # )
                        current_entities = torch.tensor(
                            episode.current_entities,  # ✅ 使用episode的真实位置
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
                    
                    # Calculate rewards
                    rewards = episode.get_reward()
                    
                    # # Calculate metrics
                    # reward_reshape = rewards.reshape((-1, self.test_rollouts))
                    # reward_reshape = np.sum(reward_reshape, axis=1)
                    
                    # reward_1 = (reward_reshape == self.positive_reward)
                    # reward_3 = (reward_reshape >= self.positive_reward * 0.33)
                    # reward_10 = (reward_reshape >= self.positive_reward * 0.1)
                    # reward_20 = (reward_reshape >= self.positive_reward * 0.05)                    
                    # all_final_reward_1 += np.sum(reward_1)
                    # all_final_reward_3 += np.sum(reward_3)
                    # all_final_reward_10 += np.sum(reward_10)
                    # all_final_reward_20 += np.sum(reward_20)
                    
                    # # Calculate MRR (simplified)
                    # for i in range(len(reward_reshape)):
                    #     if reward_reshape[i] > 0:
                    #         rank = 1  # Simplified ranking
                    #         auc += 1.0 / rank
                    # ✅ 正确的精确 metrics 计算
                    # Calculate metrics
                    rewards = episode.get_reward()  # shape: [num_examples * test_rollouts]
                    # final_entities = state['current_entities']  # shape: [num_examples * test_rollouts]
                    final_entities = episode.current_entities
                    target_entities = episode.end_entities  # shape: [num_examples * test_rollouts]

                    # Reshape 为 [num_examples, test_rollouts]
                    num_examples = len(rewards) // self.test_rollouts
                    reward_reshape = rewards.reshape((num_examples, self.test_rollouts))
                    entities_reshape = final_entities.reshape((num_examples, self.test_rollouts))
                    targets_reshape = target_entities.reshape((num_examples, self.test_rollouts))

                    # 对每个测试样本计算排名
                    from collections import Counter

                    # for i in range(num_examples):
                    #     sample_entities = entities_reshape[i]  # [test_rollouts]
                    #     target_entity = targets_reshape[i][0]  # 所有 rollouts 的目标相同，取第一个
                        
                    #     # 统计每个实体出现的次数（作为置信度）
                    #     entity_counts = Counter(sample_entities)
                        
                    #     # 按出现次数排序（次数越多，排名越靠前）
                    #     sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
                    #     ranked_entity_ids = [entity_id for entity_id, count in sorted_entities]
                    for i in range(num_examples):
                        sample_entities = entities_reshape[i]  # [test_rollouts]
                        sample_rewards = reward_reshape[i]     # [test_rollouts]
                        target_entity = targets_reshape[i][0]
                        
                        # ✅ 使用奖励加权的排名
                        from collections import defaultdict
                        entity_scores = defaultdict(float)
                        
                        for j, entity_id in enumerate(sample_entities):
                            reward = sample_rewards[j]
                            
                            # 如果这条路径成功到达目标，给予更高权重
                            if reward > 0:
                                entity_scores[entity_id] += 100.0  # 成功路径权重非常高
                            else:
                                entity_scores[entity_id] += 1.0     # 失败路径权重低
                        
                        # 按加权得分排序
                        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
                        ranked_entity_ids = [entity_id for entity_id, score in sorted_entities]
                        
                        # 找到目标实体的排名
                        if target_entity in ranked_entity_ids:
                            rank = ranked_entity_ids.index(target_entity) + 1  # 排名从1开始
                        else:
                            rank = len(ranked_entity_ids) + 1  # 如果不在列表中，给最差排名
                        
                        # 累加各项指标
                        if rank <= 1:
                            all_final_reward_1 += 1
                        if rank <= 3:
                            all_final_reward_3 += 1
                        if rank <= 10:
                            all_final_reward_10 += 1
                        if rank <= 20:
                            all_final_reward_20 += 1
                        
                        # MRR (Mean Reciprocal Rank)
                        auc += 1.0 / rank
        
        self.agent.train()
        
        # Normalize metrics
        all_final_reward_1 = all_final_reward_1 / total_examples
        all_final_reward_3 = all_final_reward_3 / total_examples
        all_final_reward_10 = all_final_reward_10 / total_examples
        all_final_reward_20 = all_final_reward_20 / total_examples
        auc = auc / total_examples
        
        # Log results
        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("MRR: {0:7.4f}".format(auc))
        
        # Save to file
        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}\n".format(all_final_reward_1))
            score_file.write("Hits@3: {0:7.4f}\n".format(all_final_reward_3))
            score_file.write("Hits@10: {0:7.4f}\n".format(all_final_reward_10))
            score_file.write("Hits@20: {0:7.4f}\n".format(all_final_reward_20))
            score_file.write("MRR: {0:7.4f}\n".format(auc))
            score_file.write("\n")
        
        return all_final_reward_10
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_counter': self.batch_counter,
            'baseline': self.baseline.b,
        }, path)
        logger.info("Model saved to: " + path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    
    # Create trainer
    trainer = Trainer(options)
    
    if not options['load_model']:
        trainer.initialize_pretrained_embeddings()
        trainer.train()
    else:
        trainer.load_model(options['model_load_dir'])
        trainer.test(beam=True, print_paths=True, save_model=False)
