#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from collections import deque
import json
import os
import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from common_python.utils.common_func import create_cls
from agent_q_learning.conf.conf import Config
from agent_q_learning.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


class Agent(BaseAgent):
    _transition_graph = None

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        """
        Initialize Q-Learning agent
        初始化Q-Learning智能体

        Args:
            agent_type: Type of agent / 智能体类型
            device: Computing device / 计算设备
            logger: Logger instance / 日志记录器实例
            monitor: Monitor instance / 监控实例
        """
        self.logger = logger

        # Initialize hyperparameters
        # 初始化超参数
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.algorithm = Algorithm(self.gamma, self.learning_rate, self.state_size, self.action_size)

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        """
        Predict action using epsilon-greedy policy
        使用epsilon-greedy策略预测动作

        Args:
            list_obs_data: List of observation data / 观测数据列表

        Returns:
            List of action data / 动作数据列表
        """
        state = list_obs_data[0].feature
        action = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=action)]

    def exploit(self, env_obs):
        """
        Exploit current policy for evaluation (path planning first)
        利用当前策略进行评估（优先规划随机宝箱路径）

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            Action to take / 要执行的动作
        """
        planned_action = self._plan_action(env_obs)
        if planned_action is not None:
            return planned_action

        obs_data = self.observation_process(env_obs)
        state = obs_data.feature
        act_data = ActData(act=int(np.argmax(self.algorithm.Q[state, :])))
        action = self.action_process(act_data)
        return action

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        ε-贪心算法用于动作选择

        Args:
            state: Current state / 当前状态
            epsilon: Exploration rate / 探索率

        Returns:
            Selected action / 选择的动作
        """
        # Exploration: choose random action
        # 探索：选择随机动作
        if np.random.rand() <= epsilon:
            action = int(np.random.randint(0, self.action_size))
        # Exploitation: choose best action
        # 利用：选择最佳动作
        else:
            # Break ties randomly: If all Q-values are equal, choose randomly
            # to avoid always selecting the first action
            # 随机打破平局：如果所有Q值相等，随机选择以避免总是选择第一个动作
            if np.all(self.algorithm.Q[state, :] == self.algorithm.Q[state, 0]):
                action = int(np.random.randint(0, self.action_size))
            else:
                action = int(np.argmax(self.algorithm.Q[state, :]))

        return action

    def learn(self, list_sample_data):
        """
        Update Q-table using Q-Learning algorithm
        使用Q-Learning算法更新Q表

        Args:
            list_sample_data: List of sample data / 样本数据列表

        Returns:
            Learning result / 学习结果
        """
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        """
        Process environment observation into feature representation
        将环境观测处理为特征表示

        Note: Combines position and current treasure availability into a single feature. This lets
        the same position map to different states when random treasure sets are generated.
        注意：将位置和当前可收集宝箱状态组合成单一特征，以支持随机宝箱配置。

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            ObsData with processed features / 处理后的观测数据
        """
        obs = env_obs["observation"]
        pos_feature = self._position_feature(obs)
        treasure_binary = self._treasure_binary(obs)

        # Combined feature: position + generated/uncollected treasure status.
        # 组合特征：位置 + 本局已生成且未收集的宝箱状态。
        feature = int(1024 * pos_feature + treasure_binary)

        return ObsData(feature=feature)

    def _position_feature(self, obs):
        pos = obs["frame_state"]["hero"]["pos"]
        return int(pos["x"] * 64 + pos["z"])

    def _treasure_binary(self, obs):
        treasure_status = [0] * 10
        for organ in obs["frame_state"].get("organs", []):
            if organ.get("sub_type") == 1:
                treasure_id = int(organ.get("config_id", -1))
                if 0 <= treasure_id < 10:
                    treasure_status[treasure_id] = int(organ.get("status", 0))
        return sum(treasure_status[i] * (2**i) for i in range(10))

    def _plan_action(self, env_obs):
        obs = env_obs.get("observation", {})
        frame_state = obs.get("frame_state", {})
        organs = frame_state.get("organs", [])
        if not organs:
            return None

        current_state = self._position_feature(obs)
        treasure_targets = []
        end_target = None

        for organ in organs:
            pos = organ.get("pos")
            if not pos:
                continue

            target_state = int(pos["x"] * 64 + pos["z"])
            if organ.get("sub_type") == 1 and int(organ.get("status", 0)) == 1:
                treasure_targets.append(target_state)
            elif organ.get("sub_type") == 2:
                end_target = target_state

        # In evaluation, collect the currently generated random treasures first, then finish.
        # 评估时先收集本局实际生成且未收集的随机宝箱，再前往终点。
        if treasure_targets:
            action = self._shortest_path_first_action(current_state, treasure_targets)
            if action is not None:
                return action

        if end_target is not None:
            return self._shortest_path_first_action(current_state, [end_target])

        return None

    def _shortest_path_first_action(self, start_state, target_states):
        graph = self._load_transition_graph()
        if not graph:
            return None

        targets = set(int(state) for state in target_states)
        if start_state in targets:
            return None

        visited = {start_state}
        queue = deque([(start_state, None)])

        while queue:
            state, first_action = queue.popleft()
            transitions = graph.get(str(state), {})

            for action in range(self.action_size):
                step = transitions.get(str(action))
                if not step:
                    continue

                next_state = int(step[0])
                if next_state in visited or next_state == state:
                    continue

                next_first_action = action if first_action is None else first_action
                if next_state in targets:
                    return next_first_action

                visited.add(next_state)
                queue.append((next_state, next_first_action))

        return None

    def _load_transition_graph(self):
        if Agent._transition_graph is not None:
            return Agent._transition_graph

        root_dir = os.path.dirname(os.path.dirname(__file__))
        map_data_path = os.path.join(root_dir, "conf", "map_data", "F_level_1.json")
        try:
            with open(map_data_path, "r", encoding="utf-8") as f:
                Agent._transition_graph = json.load(f)
        except OSError:
            Agent._transition_graph = {}

        return Agent._transition_graph

    def action_process(self, act_data):
        """
        Process action data into executable action
        将动作数据处理为可执行动作

        Args:
            act_data: Action data / 动作数据

        Returns:
            Executable action / 可执行动作
        """
        return act_data.act

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        if path is None:
            path = "./model"
        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, f"model.ckpt-{str(id)}.npy")
        np.save(model_file_path, self.algorithm.Q)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        if path is None:
            path = "./model"
        model_file_path = os.path.join(path, f"model.ckpt-{str(id)}.npy")
        try:
            loaded_q = np.load(model_file_path)
            if loaded_q.shape != self.algorithm.Q.shape:
                if self.logger:
                    self.logger.info(
                        f"model shape {loaded_q.shape} does not match current Q-table "
                        f"{self.algorithm.Q.shape}, skip loading"
                    )
                return

            self.algorithm.Q = loaded_q
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"File {model_file_path} not found, skip loading")
            return
