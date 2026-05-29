#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
"""


class Config:
    # 局部地图视野窗口大小，必须为 [3, 21] 内的奇数。
    VIEW_SIZE = 11
    if VIEW_SIZE < 3 or VIEW_SIZE > 21:
        raise ValueError(f"Config.VIEW_SIZE must be in [3, 21], got {VIEW_SIZE}")
    if VIEW_SIZE % 2 == 0:
        raise ValueError(f"Config.VIEW_SIZE must be odd, got {VIEW_SIZE}")

    VIEW_RADIUS = VIEW_SIZE // 2
    MAP_FEAT_DIM = VIEW_SIZE * VIEW_SIZE

    # 运行时配置默认值。
    # 优先从 env_info 读取，这里仅作为 fallback，避免预处理器里写死实验配置。
    DEFAULT_TREASURE_COUNT = 10
    DEFAULT_BUFF_COUNT = 2
    DEFAULT_BUFF_COOLDOWN = 100
    DEFAULT_TALENT_COOLDOWN = 100
    DEFAULT_MONSTER_INTERVAL = 500
    DEFAULT_MONSTER_SPEEDUP = 700
    DEFAULT_MAX_STEP = 1000

    # 特征维度。
    # 总维度 = 102 + VIEW_SIZE^2，由 Config 动态决定。
    FEATURES = [
        6,              # hero_core_block
        4,              # flash_stage_global_block
        10,             # monster_block
        8,              # resource_radar_block
        8,              # topology_block
        48,             # flash_direction_block
        16,             # legal_action_block
        2,              # progress_block
        MAP_FEAT_DIM,   # map_view_block
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # 动作空间：8 个移动方向 + 8 个闪现方向。
    ACTION_NUM = 16

    # 价值头。
    VALUE_NUM = 1

    # PPO 超参数。
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # 非 flash 奖励项 lambda，便于统一调参。
    LAMBDA_SURVIVE = 1.0
    LAMBDA_DANGER = 1.0
    LAMBDA_ESCAPE = 1.0
    LAMBDA_TREASURE = 1.0
    LAMBDA_BUFF = 1.0
    LAMBDA_ALIGN = 1.0
    LAMBDA_WALL = 1.0
    # 与「N 步前位置」过近时惩罚（防止原地打转 / 小范围兜圈），见 Preprocessor 中 r_stasis。
    LAMBDA_STASIS = 1.0
    LAMBDA_FINAL_FAIL = 1.0
    LAMBDA_FINAL_WIN = 1.0

    LAMBDA_EXPLORATION_NEW_CELL_COEF = 0.02

    # 长窗口位移过小惩罚：当前位置与 N 步前欧氏距离小于阈值时施加（线性，越近越罚）。
    HERO_STASIS_LOOKBACK_STEPS = 10
    HERO_STASIS_DIST_MAX = 5.0
    HERO_STASIS_PENALTY = 1.0

    # Flash v1 奖励项 lambda。
    LAMBDA_FLASH_ESCAPE = 1.0
    LAMBDA_FLASH_ABUSE = 1.0
    LAMBDA_FLASH_PATH_RESOURCE = 1.0
    LAMBDA_FLASH_LANDING = 1.0
    LAMBDA_FLASH_CROSS_WALL = 1.0
    LAMBDA_FLASH_WALL_HIT = 1.0
    LAMBDA_FLASH_DANGER_FLEE = 1.0
    LAMBDA_FLASH_PAST_MONSTER = 1.0

    # Flash 特征与奖励内部超参数。
    # 这些不是 reward lambda，而是 flash 候选特征 / reward 计算内部权重。
    FLASH_PATH_TREASURE_WEIGHT = 1.0
    FLASH_PATH_BUFF_WEIGHT = 0.5

    FLASH_LANDING_OPEN_WEIGHT = 1.0
    FLASH_LANDING_DEADEND_WEIGHT = 1.0

    PRESSURE_W_DANGER = 0.45
    PRESSURE_W_MONSTER2 = 0.25
    PRESSURE_W_SPEEDUP = 0.15
    PRESSURE_W_TRAP = 0.15

    FLASH_ABUSE_SAFE_GAIN_THR = 0.05
    FLASH_ABUSE_RESOURCE_THR = 1e-6
    FLASH_ABUSE_OPEN_GAIN_THR = 0.0

    FLASH_CROSS_MIN_REACHABLE_RATIO = 0.35

    # 闪现撞墙：上、下、左、右 10 格、斜向 8 格；仅当能落在「整段射程末端」时才算未撞墙。
    # 实际落点与「上一刻位置 + 方向 * 整段格数」不一致，说明目标格或路径上存在障碍、被退回，判为撞墙（仅数值容差）。
    FLASH_WALL_POS_MATCH_TOL = 0.35
    FLASH_WALL_HIT_MAG = 0.2

    # 附近怪物时：穿墙拉开距离、路径掠过怪物，额外 shaping（与 FLASH_MONSTER_* 同单位：世界距离）。
    FLASH_DANGER_FLEE_MONSTER_DIST = 22.0
    FLASH_DANGER_FLEE_MIN_GAIN = 0.4
    FLASH_DANGER_FLEE_GAIN_NORM = 8.0

    FLASH_PAST_MONSTER_PATH_DIST = 2.5
    FLASH_PAST_MONSTER_MIN_GAIN = 0.2
    FLASH_PAST_MONSTER_REWARD = 0.12

    # 默认用落点周围 5x5 小窗口估算开阔度。
    FLASH_LANDING_WINDOW_RADIUS = 2
