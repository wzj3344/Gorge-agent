#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
"""

import math

import numpy as np

from agent_ppo.conf.conf import Config

MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 500.0
MAP_SIZE = 128.0
MAP_SIDE = 128
VISION_HALF = 10  # 21×21 视野：中心向各方向 10 格
FLASH_RANGE = np.array([10.0, 8.0, 10.0, 8.0, 10.0, 8.0, 10.0, 8.0], dtype=np.float32)

# 方向顺序与动作严格对齐：右、右上、上、左上、左、左下、下、右下
FLASH_DIR_RC = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
FLASH_DIR_XZ = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]


DIR_MAP = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.707, 0.707),
    3: (0.0, 1.0),
    4: (-0.707, 0.707),
    5: (-1.0, 0.0),
    6: (-0.707, -0.707),
    7: (0.0, -1.0),
    8: (0.707, -0.707),
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0



def _update_vision_and_count_new(unexplored, hx, hz):
    """将以 (hx,hz) 为中心的 (2*VISION_HALF+1)^2 视野标为已探索(0)，返回本步新揭示格子数。"""
    new_count = 0
    x0 = max(0, hx - VISION_HALF)
    x1 = min(MAP_SIDE - 1, hx + VISION_HALF)
    z0 = max(0, hz - VISION_HALF)
    z1 = min(MAP_SIDE - 1, hz + VISION_HALF)
    for gx in range(x0, x1 + 1):
        for gz in range(z0, z1 + 1):
            if unexplored[gx, gz] != 0:
                new_count += 1
                unexplored[gx, gz] = 0
    return new_count


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = Config.DEFAULT_MAX_STEP
        self.last_min_monster_dist_raw_visible = None
        self.last_min_monster_dist_raw_all = None
        self.last_hero_pos = None
        self.last_score = 0.0
        self.last_buff_time = 0.0
        self.last_treasures = 0
        self.last_buffs = 0
        self.prev_flash_dir_reachable_ratio = np.zeros(8, dtype=np.float32)
        self.prev_flash_dir_cross_wall_flag = np.zeros(8, dtype=np.float32)
        self.prev_flash_dir_actual_range = np.ones(8, dtype=np.float32)
        self.prev_survive_pressure_score = 0.0
        self.prev_local_open_score = 0.0
        self.prev_selected_flash_range = 1.0

        # 全局探索图：1=从未进入过视野，0=曾进入过视野
        self._unexplored = np.ones((MAP_SIDE, MAP_SIDE), dtype=np.uint8)
        # 本局内每步 (x, z)，用于与「N 步前」比较是否长时间滞留在小范围
        self._hero_pos_history = []

    def _safe_int(self, value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _pick_runtime_value(self, env_info, keys, default, positive_only=False):
        value = default
        for key in keys:
            if key in env_info and env_info[key] is not None:
                value = env_info[key]
                break
        try:
            value = int(value) if isinstance(default, int) else float(value)
        except (TypeError, ValueError):
            value = default
        if positive_only and value <= 0:
            return default
        return value

    def _get_runtime_cfg(self, env_info):
        return {
            "treasure_count": self._pick_runtime_value(env_info, ["treasure_count", "total_treasure"], Config.DEFAULT_TREASURE_COUNT),
            "buff_count": self._pick_runtime_value(env_info, ["buff_count", "total_buff"], Config.DEFAULT_BUFF_COUNT),
            "buff_cooldown": self._pick_runtime_value(env_info, ["buff_cooldown"], Config.DEFAULT_BUFF_COOLDOWN, positive_only=True),
            "talent_cooldown": self._pick_runtime_value(env_info, ["talent_cooldown", "flash_cooldown"], Config.DEFAULT_TALENT_COOLDOWN, positive_only=True),
            "monster_interval": self._pick_runtime_value(env_info, ["monster_interval"], Config.DEFAULT_MONSTER_INTERVAL, positive_only=True),
            "monster_speedup": self._pick_runtime_value(env_info, ["monster_speedup"], Config.DEFAULT_MONSTER_SPEEDUP, positive_only=True),
            "max_step": self._pick_runtime_value(env_info, ["max_step"], Config.DEFAULT_MAX_STEP, positive_only=True),
        }

    def _is_passable(self, map_info, row, col):
        return (
            map_info is not None
            and row >= 0
            and col >= 0
            and row < len(map_info)
            and col < len(map_info[0])
            and map_info[row][col] != 0
        )

    def _estimate_bucket_distance(self, bucket):
        return float(bucket) * 30.0 + 15.0

    def _distance_pos(self, pos_a, pos_b):
        return math.sqrt((pos_a["x"] - pos_b["x"]) ** 2 + (pos_a["z"] - pos_b["z"]) ** 2)

    def _estimate_entity_pos_from_relative(self, hero_pos, direction, bucket):
        dir_vec = DIR_MAP.get(direction, (0.0, 0.0))
        est_dist = self._estimate_bucket_distance(bucket)
        return {"x": hero_pos["x"] + dir_vec[0] * est_dist, "z": hero_pos["z"] + dir_vec[1] * est_dist}

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        sx, sz = seg_end["x"] - seg_start["x"], seg_end["z"] - seg_start["z"]
        seg_len_sq = sx * sx + sz * sz
        if seg_len_sq <= 1e-6:
            return self._distance_pos(point, seg_start)
        px, pz = point["x"] - seg_start["x"], point["z"] - seg_start["z"]
        proj = float(np.clip((px * sx + pz * sz) / seg_len_sq, 0.0, 1.0))
        closest = {"x": seg_start["x"] + proj * sx, "z": seg_start["z"] + proj * sz}
        return self._distance_pos(point, closest)

    def _find_flash_landing(self, map_info, center, dir_rc, flash_range):
        if map_info is None or center is None:
            return 0, center, False
        dr, dc = dir_rc
        landing_step, landing_cell = 0, center
        for step in range(int(flash_range), 0, -1):
            row, col = center[0] + dr * step, center[1] + dc * step
            if self._is_passable(map_info, row, col):
                landing_step, landing_cell = step, (row, col)
                break
        crossed_wall = False
        for step in range(1, landing_step):
            row, col = center[0] + dr * step, center[1] + dc * step
            if not self._is_passable(map_info, row, col):
                crossed_wall = True
                break
        return landing_step, landing_cell, crossed_wall

    def _compute_landing_open_score(self, map_info, landing_cell):
        if map_info is None or landing_cell is None:
            return 0.0
        radius = Config.FLASH_LANDING_WINDOW_RADIUS
        total_count = float((2 * radius + 1) ** 2)
        passable_count = 0.0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if self._is_passable(map_info, landing_cell[0] + dr, landing_cell[1] + dc):
                    passable_count += 1.0
        return float(passable_count / max(total_count, 1.0))

    def _count_resources_on_flash_path(self, organs, hero_pos, landing_pos, treasure_count_cfg, buff_count_cfg):
        if organs is None:
            return 0.0
        n_treasure, n_buff = 0.0, 0.0
        for organ in organs:
            if organ.get("status", 1) != 1:
                continue
            organ_pos = organ.get("pos")
            if not isinstance(organ_pos, dict) or "x" not in organ_pos or "z" not in organ_pos:
                organ_pos = self._estimate_entity_pos_from_relative(
                    hero_pos, organ.get("hero_relative_direction", 0), organ.get("hero_l2_distance", 5)
                )
            if self._point_to_segment_distance(organ_pos, hero_pos, landing_pos) > 1.25:
                continue
            if organ.get("sub_type") == 1:
                n_treasure += 1.0
            elif organ.get("sub_type") == 2:
                n_buff += 1.0
        return float(
            Config.FLASH_PATH_TREASURE_WEIGHT * n_treasure / max(float(treasure_count_cfg), 1.0)
            + Config.FLASH_PATH_BUFF_WEIGHT * n_buff / max(float(buff_count_cfg), 1.0)
        )

    def _compute_flash_direction_features(self, map_info, hero_pos, monster_positions, cur_min_dist_raw_all, organs, runtime_cfg):
        """Flash v1 候选特征：6 组 * 8 个方向。"""
        reachable_ratio = np.zeros(8, dtype=np.float32)
        cross_wall_flag = np.zeros(8, dtype=np.float32)
        landing_open_score = np.zeros(8, dtype=np.float32)
        landing_dead_end_risk = np.zeros(8, dtype=np.float32)
        safe_gain = np.zeros(8, dtype=np.float32)
        path_resource_gain = np.zeros(8, dtype=np.float32)
        actual_range = np.ones(8, dtype=np.float32)

        center = None
        if map_info is not None and len(map_info) > 0 and len(map_info[0]) > 0:
            center = (len(map_info) // 2, len(map_info[0]) // 2)

        for idx in range(8):
            flash_range = FLASH_RANGE[idx]
            landing_step, landing_cell, crossed_wall = self._find_flash_landing(map_info, center, FLASH_DIR_RC[idx], flash_range)
            reachable_ratio[idx] = float(landing_step / max(flash_range, 1.0))
            cross_wall_flag[idx] = float(crossed_wall and reachable_ratio[idx] >= Config.FLASH_CROSS_MIN_REACHABLE_RATIO)
            actual_range[idx] = float(max(landing_step, 1))

            landing_open = self._compute_landing_open_score(map_info, landing_cell)
            landing_open_score[idx] = float(landing_open)
            landing_dead_end_risk[idx] = float(1.0 - landing_open)

            dir_x, dir_z = FLASH_DIR_XZ[idx]
            landing_pos = {"x": hero_pos["x"] + dir_x * landing_step, "z": hero_pos["z"] + dir_z * landing_step}
            d_after = cur_min_dist_raw_all
            if monster_positions:
                d_after = min(self._distance_pos(landing_pos, monster_pos) for monster_pos in monster_positions)
            safe_gain[idx] = float(np.clip((d_after - cur_min_dist_raw_all) / max(flash_range, 1.0), -1.0, 1.0))
            path_resource_gain[idx] = self._count_resources_on_flash_path(
                organs, hero_pos, landing_pos, runtime_cfg["treasure_count"], runtime_cfg["buff_count"]
            )

        return {
            "feature": np.concatenate(
                [
                    reachable_ratio,
                    cross_wall_flag,
                    landing_open_score,
                    landing_dead_end_risk,
                    safe_gain,
                    path_resource_gain,
                ]
            ).astype(np.float32),
            "reachable_ratio": reachable_ratio,
            "cross_wall_flag": cross_wall_flag,
            "actual_range": actual_range,
        }

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]
        runtime_cfg = self._get_runtime_cfg(env_info)
        self.step_no = observation["step_no"]
        self.max_step = runtime_cfg["max_step"]
        last_action = self._safe_int(last_action, -1)

        # 1) hero_core_block
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        flash_cd = hero.get("flash_cooldown", 0)
        buff_time = hero.get("buff_remaining_time", 0)
        hero_feat = np.array(
            [
                _norm(hero_pos["x"], MAP_SIZE),
                _norm(hero_pos["z"], MAP_SIZE),
                _norm(flash_cd, MAX_FLASH_CD),
                _norm(buff_time, MAX_BUFF_DURATION),
                1.0 if flash_cd == 0 else 0.0,
                1.0 if buff_time > 0 else 0.0,
            ],
            dtype=np.float32,
        )

        # 2) monster_block，同时维护 visible / all 两套最近距离
        monsters = frame_state.get("monsters", [])
        monster_feats, monster_positions, visible_raw_dists, all_raw_dists = [], [], [], []
        for idx in range(2):
            if idx < len(monsters):
                monster = monsters[idx]
                is_in_view = float(monster.get("is_in_view", 0))
                m_speed_norm = _norm(monster.get("speed", 1), MAX_MONSTER_SPEED)
                if is_in_view:
                    monster_pos = monster["pos"]
                    dx, dz = monster_pos["x"] - hero_pos["x"], monster_pos["z"] - hero_pos["z"]
                    raw_dist = math.sqrt(dx**2 + dz**2)
                    dx_norm, dz_norm = np.clip(dx / 40.0, -1.0, 1.0), np.clip(dz / 40.0, -1.0, 1.0)
                    visible_raw_dists.append(raw_dist)
                else:
                    direction = monster.get("hero_relative_direction", 0)
                    bucket = monster.get("hero_l2_distance", 5)
                    monster_pos = self._estimate_entity_pos_from_relative(hero_pos, direction, bucket)
                    raw_dist = self._estimate_bucket_distance(bucket)
                    dir_vec = DIR_MAP.get(direction, (0.0, 0.0))
                    dx_norm, dz_norm = dir_vec[0], dir_vec[1]
                dist_norm = np.clip(raw_dist / 40.0, 0.0, 1.0)
                monster_feats.append(np.array([is_in_view, dx_norm, dz_norm, m_speed_norm, dist_norm], dtype=np.float32))
                monster_positions.append(monster_pos)
                all_raw_dists.append(raw_dist)
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))
        cur_min_dist_raw_visible = min(visible_raw_dists) if visible_raw_dists else 100.0
        cur_min_dist_raw_all = min(all_raw_dists) if all_raw_dists else 100.0

        # 3) resource_radar_block
        organs = frame_state.get("organs", [])
        treasure_exist, t_dx_norm, t_dz_norm, t_dist_bucket, t_direction, min_t_bucket = 0.0, 0.0, 0.0, 5.0, 0, 999
        buff_exist, b_dx_norm, b_dz_norm, b_dist_bucket, min_b_bucket = 0.0, 0.0, 0.0, 5.0, 999
        for organ in organs:
            if organ.get("status", 1) != 1:
                continue
            direction = organ.get("hero_relative_direction", 0)
            bucket = organ.get("hero_l2_distance", 5)
            dir_vec = DIR_MAP.get(direction, (0.0, 0.0))
            if organ.get("sub_type") == 1 and bucket < min_t_bucket:
                treasure_exist, min_t_bucket = 1.0, bucket
                t_direction, t_dx_norm, t_dz_norm, t_dist_bucket = direction, dir_vec[0], dir_vec[1], bucket
            elif organ.get("sub_type") == 2 and bucket < min_b_bucket:
                buff_exist, min_b_bucket = 1.0, bucket
                b_dx_norm, b_dz_norm, b_dist_bucket = dir_vec[0], dir_vec[1], bucket
        treasure_feat = np.array([treasure_exist, t_dx_norm, t_dz_norm, t_dist_bucket / 5.0 if treasure_exist > 0 else 1.0], dtype=np.float32)
        buff_feat = np.array([buff_exist, b_dx_norm, b_dz_norm, b_dist_bucket / 5.0 if buff_exist > 0 else 1.0], dtype=np.float32)

        # 4) topology_block
        ray_feat = np.zeros(8, dtype=np.float32)
        if map_info is not None and len(map_info) > 0 and len(map_info[0]) > 0:
            center = len(map_info) // 2
            for idx, (dr, dc) in enumerate(FLASH_DIR_RC):
                row, col, dist = center, center, 0.0
                for _ in range(10):
                    row += dr
                    col += dc
                    if self._is_passable(map_info, row, col):
                        dist += 1.0
                    else:
                        break
                ray_feat[idx] = dist / 10.0
        current_local_open_score = float(np.mean(ray_feat))
        current_local_dead_end_risk = float(1.0 - current_local_open_score)

        # 5) flash_stage_global_block
        flash_cd_ratio_cfg = float(np.clip(flash_cd / max(float(runtime_cfg["talent_cooldown"]), 1.0), 0.0, 1.0))
        time_to_monster2_ratio = float(max(float(runtime_cfg["monster_interval"]) - float(self.step_no), 0.0) / max(float(runtime_cfg["monster_interval"]), 1.0))
        time_to_monster_speedup_ratio = float(max(float(runtime_cfg["monster_speedup"]) - float(self.step_no), 0.0) / max(float(runtime_cfg["monster_speedup"]), 1.0))
        danger_term = float(np.clip(1.0 - cur_min_dist_raw_all / 10.0, 0.0, 1.0))
        monster2_term = float(1.0 - time_to_monster2_ratio)
        speedup_term = float(1.0 - time_to_monster_speedup_ratio)
        trap_term = float(np.clip(1.0 - current_local_open_score, 0.0, 1.0))
        survive_pressure_score = float(
            np.clip(
                Config.PRESSURE_W_DANGER * danger_term
                + Config.PRESSURE_W_MONSTER2 * monster2_term
                + Config.PRESSURE_W_SPEEDUP * speedup_term
                + Config.PRESSURE_W_TRAP * trap_term,
                0.0,
                1.0,
            )
        )
        flash_stage_global_feat = np.array(
            [flash_cd_ratio_cfg, time_to_monster2_ratio, time_to_monster_speedup_ratio, survive_pressure_score],
            dtype=np.float32,
        )

        # 6) flash_direction_block
        flash_direction_info = self._compute_flash_direction_features(
            map_info, hero_pos, monster_positions, cur_min_dist_raw_all, organs, runtime_cfg
        )
        flash_direction_feat = flash_direction_info["feature"]

        # 7) legal_action_block
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for idx in range(min(16, len(legal_act_raw))):
                    legal_action[idx] = int(legal_act_raw[idx])
            else:
                valid_set = {int(action) for action in legal_act_raw if int(action) < 16}
                legal_action = [1 if idx in valid_set else 0 for idx in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # 8) progress_block
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, step_norm], dtype=np.float32)

        # 9) map_view_block，放在输入最后
        map_feat = np.zeros(Config.MAP_FEAT_DIM, dtype=np.float32)
        try:
            if map_info is not None and len(map_info) > 0 and len(map_info[0]) > 0:
                center_row, center_col = len(map_info) // 2, len(map_info[0]) // 2
                radius, flat_idx = Config.VIEW_RADIUS, 0
                for row in range(center_row - radius, center_row + radius + 1):
                    for col in range(center_col - radius, center_col + radius + 1):
                        if self._is_passable(map_info, row, col):
                            map_feat[flat_idx] = 1.0
                        flat_idx += 1
        except (TypeError, IndexError, KeyError):
            map_feat = np.zeros(Config.MAP_FEAT_DIM, dtype=np.float32)


        # 8. 全局探索：更新 128×128 视野覆盖
        hx_i = int(np.clip(round(hero_pos["x"]), 0, MAP_SIDE - 1))
        hz_i = int(np.clip(round(hero_pos["z"]), 0, MAP_SIDE - 1))
        new_cells_in_vision = _update_vision_and_count_new(self._unexplored, hx_i, hz_i)

        # Total feature dim = 102 + VIEW_SIZE^2
        feature = np.concatenate(
            [
                hero_feat,
                flash_stage_global_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                ray_feat,
                flash_direction_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                map_feat,
            ]
        )

        # -------------------- Reward: 逐项定义 -> 统一汇总 --------------------
        prev_min_dist_visible = self.last_min_monster_dist_raw_visible
        prev_min_dist_all = self.last_min_monster_dist_raw_all
        last_treasures_before_update = self.last_treasures
        last_buffs_before_update = self.last_buffs
        current_treasures = env_info.get("treasures_collected", 0)
        current_buffs = env_info.get("collected_buff", 0)

        # 基础生存分
        r_survive = 0.01

        # visible-only 怪物危险惩罚
        #r_danger = -0.5 if cur_min_dist_raw_visible < 3.0 else (-0.2 if cur_min_dist_raw_visible < 6.0 else 0.0)
        if(cur_min_dist_raw_visible < 15.0):
            r_danger= 0.1 * (cur_min_dist_raw_visible - 15.0)
        else:
            r_danger=0

        # visible-only 逃生 shaping
        r_escape = 0.0
        if prev_min_dist_visible is not None and cur_min_dist_raw_visible < 25.0:
            r_escape = float(np.clip(cur_min_dist_raw_visible - prev_min_dist_visible, -3.0, 3.0) * 0.05)

        # 资源奖励
        r_treasure = 20.0 if current_treasures > last_treasures_before_update else 0.0
        r_buff = 10.0 if current_buffs > last_buffs_before_update else 0.0

        # 朝最近宝箱方向位移引导
        r_align = 0.0
        if treasure_exist > 0 and self.last_hero_pos is not None:
            h_dx, h_dz = hero_pos["x"] - self.last_hero_pos["x"], hero_pos["z"] - self.last_hero_pos["z"]
            target_dir = DIR_MAP.get(t_direction, (0.0, 0.0))
            r_align = float(np.clip(h_dx * target_dir[0] + h_dz * target_dir[1], -2.0, 2.0) * 0.05)

        # 普通移动撞墙惩罚
        r_wall = 0.0
        if self.last_hero_pos is not None:
            dx, dz = hero_pos["x"] - self.last_hero_pos["x"], hero_pos["z"] - self.last_hero_pos["z"]
            if dx == 0 and dz == 0 and 0 <= last_action <= 7:
                r_wall = -0.2

        # 长窗口内净位移过小惩罚（N 步前与当前欧氏距过小 → 可能原地打转 / 小范围震荡）
        r_stasis = 0.0
        lookback = Config.HERO_STASIS_LOOKBACK_STEPS
        if lookback > 0 and len(self._hero_pos_history) >= lookback:
            px, pz = self._hero_pos_history[-lookback]
            dist_ago = float(math.hypot(hero_pos["x"] - px, hero_pos["z"] - pz))
            thr = float(Config.HERO_STASIS_DIST_MAX)
            if dist_ago < thr and thr > 1e-6:
                r_stasis = -Config.HERO_STASIS_PENALTY * (1.0 - dist_ago / thr)

        # Flash v1 reward：评价上一动作造成的当前结果
        r_flash_escape = 0.0
        r_flash_abuse = 0.0
        r_flash_path_resource = 0.0
        r_flash_landing = 0.0
        r_flash_cross_wall = 0.0
        r_flash_wall_hit = 0.0
        r_flash_danger_flee = 0.0
        r_flash_past_monster = 0.0

        selected_idx = last_action - 8 if 8 <= last_action <= 15 else -1
        if 0 <= selected_idx < 8:
            selected_flash_range = float(max(self.prev_flash_dir_actual_range[selected_idx], 1.0))
            self.prev_selected_flash_range = selected_flash_range
            actual_safe_gain = 0.0
            if prev_min_dist_all is not None:
                actual_safe_gain = float(
                    np.clip((cur_min_dist_raw_all - prev_min_dist_all) / max(selected_flash_range, 1.0), -1.0, 1.0)
                )
            delta_treasure = max(current_treasures - last_treasures_before_update, 0)
            delta_buff = max(current_buffs - last_buffs_before_update, 0)
            actual_resource_gain = (
                Config.FLASH_PATH_TREASURE_WEIGHT * float(delta_treasure) / max(float(runtime_cfg["treasure_count"]), 1.0)
                + Config.FLASH_PATH_BUFF_WEIGHT * float(delta_buff) / max(float(runtime_cfg["buff_count"]), 1.0)
            )
            open_gain = current_local_open_score - self.prev_local_open_score
            current_dead, prev_dead = current_local_dead_end_risk, 1.0 - self.prev_local_open_score

            # 1. 高生存压力下，flash 后实际脱险则奖励
            r_flash_escape = Config.LAMBDA_FLASH_ESCAPE * self.prev_survive_pressure_score * max(actual_safe_gain, 0.0)

            # 2. 高压力时，既没脱险也没收益也没改善站位，则惩罚滥用 flash
            abuse_flag = float(
                actual_safe_gain < Config.FLASH_ABUSE_SAFE_GAIN_THR
                and actual_resource_gain <= Config.FLASH_ABUSE_RESOURCE_THR
                and open_gain <= Config.FLASH_ABUSE_OPEN_GAIN_THR
            )
            r_flash_abuse = -Config.LAMBDA_FLASH_ABUSE * self.prev_survive_pressure_score * abuse_flag

            # 3. 低压力阶段鼓励 flash 赶路吃资源
            # r_flash_path_resource = (
            #     Config.LAMBDA_FLASH_PATH_RESOURCE * (1.0 - self.prev_survive_pressure_score) * actual_resource_gain
            # )
            r_flash_path_resource = 0

            # 4. 落点越开阔越奖励，越死胡同越惩罚
            r_flash_landing = Config.LAMBDA_FLASH_LANDING * (
                Config.FLASH_LANDING_OPEN_WEIGHT * (current_local_open_score - self.prev_local_open_score)
                - Config.FLASH_LANDING_DEADEND_WEIGHT * (current_dead - prev_dead)
            )

            # 5. 有效穿墙的弱奖励，必须乘 reachable_ratio，且不允许实际更危险
            r_flash_cross_wall = (
                Config.LAMBDA_FLASH_CROSS_WALL
                * (1.0 - self.prev_survive_pressure_score)
                * float(self.prev_flash_dir_cross_wall_flag[selected_idx])
                * float(self.prev_flash_dir_reachable_ratio[selected_idx])
                * float(actual_safe_gain >= 0.0)
            )

            if self.last_hero_pos is not None:
                r_full = int(np.round(float(FLASH_RANGE[selected_idx])))
                dx, dz = FLASH_DIR_XZ[selected_idx]
                ex = float(self.last_hero_pos["x"]) + float(dx) * float(r_full)
                ez = float(self.last_hero_pos["z"]) + float(dz) * float(r_full)
                dist_to_full_reach = float(
                    math.hypot(float(hero_pos["x"]) - ex, float(hero_pos["z"]) - ez)
                )
                if dist_to_full_reach > float(Config.FLASH_WALL_POS_MATCH_TOL):
                    r_flash_wall_hit = -float(Config.FLASH_WALL_HIT_MAG)

                # 附近有怪时，穿墙且与怪拉开距离
                d_near = float(Config.FLASH_DANGER_FLEE_MONSTER_DIST)
                d_prev = prev_min_dist_all
                d_cur = cur_min_dist_raw_all
                g_min = float(Config.FLASH_DANGER_FLEE_MIN_GAIN)
                g_norm = max(float(Config.FLASH_DANGER_FLEE_GAIN_NORM), 1e-6)
                if (
                    d_prev is not None
                    and d_prev < d_near
                    and float(self.prev_flash_dir_cross_wall_flag[selected_idx]) > 0.5
                    and (d_cur - d_prev) > g_min
                ):
                    gain = d_cur - d_prev
                    r_flash_danger_flee = float(
                        Config.LAMBDA_FLASH_DANGER_FLEE * np.clip((gain - g_min) / g_norm, 0.0, 1.0)
                    )
                # 路径掠过怪物且拉开距离（弱于上项；不依赖穿墙，可与上项同帧叠加）
                if d_prev is not None and (d_cur - d_prev) > float(Config.FLASH_PAST_MONSTER_MIN_GAIN):
                    path_thr = float(Config.FLASH_PAST_MONSTER_PATH_DIST)
                    seg_a = self.last_hero_pos
                    seg_b = {"x": hero_pos["x"], "z": hero_pos["z"]}
                    for monster_pos in monster_positions:
                        if self._point_to_segment_distance(monster_pos, seg_a, seg_b) < path_thr:
                            r_flash_past_monster = float(
                                Config.LAMBDA_FLASH_PAST_MONSTER * float(Config.FLASH_PAST_MONSTER_REWARD)
                            )
                            break
        else:
            self.prev_selected_flash_range = 1.0

        r_exploration=float(min(new_cells_in_vision,50))

        reward = (
            Config.LAMBDA_SURVIVE * r_survive
            + Config.LAMBDA_DANGER * r_danger
            + Config.LAMBDA_ESCAPE * r_escape
            + Config.LAMBDA_TREASURE * r_treasure
            + Config.LAMBDA_BUFF * r_buff
            + Config.LAMBDA_ALIGN * r_align
            + Config.LAMBDA_WALL * r_wall
            + Config.LAMBDA_STASIS * r_stasis

            + Config.LAMBDA_EXPLORATION_NEW_CELL_COEF * r_exploration

            + r_flash_escape
            + r_flash_abuse
            + r_flash_path_resource
            + r_flash_landing
            + r_flash_cross_wall
            + Config.LAMBDA_FLASH_WALL_HIT * r_flash_wall_hit
            + r_flash_danger_flee
            + r_flash_past_monster
        )

        # 统一在 reward 之后更新缓存，保证“上一动作 -> 当前结果”的对齐关系
        self.last_min_monster_dist_raw_visible = cur_min_dist_raw_visible
        self.last_min_monster_dist_raw_all = cur_min_dist_raw_all
        self.last_treasures = current_treasures
        self.last_buffs = current_buffs
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self._hero_pos_history.append((float(hero_pos["x"]), float(hero_pos["z"])))
        if len(self._hero_pos_history) > 200:
            self._hero_pos_history = self._hero_pos_history[-200:]
        self.prev_flash_dir_reachable_ratio = flash_direction_info["reachable_ratio"].copy()
        self.prev_flash_dir_cross_wall_flag = flash_direction_info["cross_wall_flag"].copy()
        self.prev_flash_dir_actual_range = flash_direction_info["actual_range"].copy()
        self.prev_survive_pressure_score = survive_pressure_score
        self.prev_local_open_score = current_local_open_score

        return feature, legal_action, [reward]
