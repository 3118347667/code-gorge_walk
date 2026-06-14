#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here.
# It must be one of dynamic_programming, monte_carlo, q_learning, sarsa or diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是dynamic_programming、monte_carlo、q_learning、sarsa、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = [
    "dynamic_programming",
    "monte_carlo",
    "q_learning",
    "sarsa",
    "diy",
]
algorithm_name = "q_learning"


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars={
            "preload_ratio": "0.2",
            "aisrv_env_ipc_method": "zmq",
            "wrapper_type": "local",
        },
        check_model_method="listdir_count",
    )
