#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


def sample_process(list_game_data):
    """
    Process game data into sample format for training
    将游戏数据处理为训练样本格式

    Args:
        list_game_data: List of game frames / 游戏帧列表

    Returns:
        List of processed samples (dict format) / 处理后的样本列表（字典格式）
    """
    return [
        {
            "state": frame.state,
            "action": frame.action,
            "reward": frame.reward,
            "next_state": frame.next_state,
            "done": frame.done,
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs, score_delta=0):
    """
    Shape reward signal for better learning
    塑形奖励信号以改善学习效果

    Args:
        env_reward: Original environment reward (unused) / 原始环境奖励（未使用）
        env_obs: Environment observation / 环境观测
        score_delta: Incremental score gained this step / 当前步新增分数

    Returns:
        Shaped reward value / 塑形后的奖励值
    """
    terminated = env_obs["terminated"]
    truncated = env_obs["truncated"]

    reward = -1

    # Goal reward.
    # 到达终点奖励。
    if terminated:
        reward += 150

    # Timeout penalty.
    # 超时截断惩罚。
    if truncated and not terminated:
        reward -= 50

    # Only add incremental environment score to avoid repeated cumulative rewards.
    # 仅加入当前步增量分数，避免重复累计奖励。
    if score_delta > 0:
        reward += score_delta

    return reward
