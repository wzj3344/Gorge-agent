#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config

def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v2"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # 策略头 (Actor Network)
        self.actor_net = nn.Sequential(
            make_fc_layer(input_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num)
        )

        # 价值头 (Critic Network)
        self.critic_net = nn.Sequential(
            make_fc_layer(input_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, value_num)
        )

    def forward(self, obs, inference=False):
        logits = self.actor_net(obs)
        value = self.critic_net(obs)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
