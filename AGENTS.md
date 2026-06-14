# Repository Guidelines

## Project Structure & Module Organization

This repository contains several reinforcement-learning agent implementations for Gorge Walk v2. Each agent package follows the same shape:

- `agent_q_learning/`, `agent_sarsa/`, `agent_monte_carlo/`, `agent_dynamic_programming/`, `agent_diy/`: algorithm-specific implementations.
- `*/agent.py`: agent lifecycle methods, prediction, feature handling, and model save/load hooks.
- `*/algorithm/algorithm.py`: learning or planning logic.
- `*/feature/`: observation and sample data definitions/preprocessing.
- `*/model/`: model persistence support.
- `*/workflow/train_workflow.py`: training workflow entry point.
- `*/conf/`: agent-specific Python and TOML configuration.
- `conf/`: shared app, algorithm, and map configuration, including `conf/map_data/`.
- `train_test.py`: local train/test launcher.

Keep changes inside the relevant agent package unless the behavior is shared across all agents.

## Build, Test, and Development Commands

- `python3 train_test.py`: runs the local train/test workflow. Set `algorithm_name` in `train_test.py` to one of `dynamic_programming`, `monte_carlo`, `q_learning`, `sarsa`, or `diy` before running.
- `python3 -m py_compile train_test.py agent_q_learning/agent.py`: quick syntax check for edited Python files.

No package manager, Makefile, or standalone test runner is defined in this checkout. The code depends on the KaiwuDRL/Tencent AI Arena runtime being available in the Python environment.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Follow the existing package conventions: snake_case modules and functions, PascalCase classes, and uppercase constants in `conf/conf.py`. Keep bilingual comments/docstrings where they already exist, and prefer short comments that explain non-obvious RL or environment assumptions. Model checkpoint files should preserve the `model.ckpt-{id}` naming pattern expected by `save_model` and `load_model`.

## Testing Guidelines

Use `train_test.py` as the primary validation path for behavior changes. For algorithm edits, run the affected agent and confirm that training completes and model files are produced or loaded as expected. For feature preprocessing changes, verify the state/action shapes still match the corresponding Q-table or model dimensions. Add focused tests only if you introduce a local test framework.

## Commit & Pull Request Guidelines

This working tree does not include Git history, so no existing commit convention can be inferred. Use concise imperative commit messages such as `Fix q_learning state encoding` or `Tune sarsa exploration config`. Pull requests should include a short summary, the affected agent(s), configuration changes, validation commands run, and screenshots or logs when training behavior changes.

## Security & Configuration Tips

Do not commit generated checkpoints, local logs, credentials, or machine-specific runtime paths. Keep environment-specific settings in TOML configuration files and document any required KaiwuDRL runtime variables in the PR description.
