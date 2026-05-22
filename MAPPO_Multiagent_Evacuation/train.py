from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from mappo import MAPPOAgent, MAPPOConfig, RolloutBuffer
from mappo_env import EvacuationConfig, MultiAgentGridEvacuationEnv
from utils import (
    choose_default_starts,
    create_demo_map,
    find_border_exits,
    load_txt_map,
    parse_points,
    write_csv,
)
from visualize import plot_paths, plot_training_curves


def build_env(args) -> MultiAgentGridEvacuationEnv:
    grid = load_txt_map(args.map_file) if args.map_file else create_demo_map()
    exits = parse_points(args.exits) or find_border_exits(grid)
    starts = parse_points(args.starts) or choose_default_starts(grid, exits, args.num_agents)
    starts = starts[: args.num_agents]
    cfg = EvacuationConfig(
        max_steps=args.max_steps,
        local_view_radius=args.local_view_radius,
        max_neighbors=args.max_neighbors,
        allow_wait=not args.no_wait,
        exit_capacity=args.exit_capacity,
    )
    return MultiAgentGridEvacuationEnv(grid, starts, exits, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO for discrete multi-agent grid evacuation.")
    parser.add_argument("--map-file", type=str, default=None, help="Optional txt map path. 0=free, 1=obstacle.")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--starts", type=str, default=None, help="Format: r,c;r,c;...")
    parser.add_argument("--exits", type=str, default=None, help="Format: r,c;r,c;...")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--local-view-radius", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--exit-capacity", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true", help="Disable the extra wait action.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = build_env(args)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir or Path(__file__).resolve().parent / "results" / run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = MAPPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    agent = MAPPOAgent(env.obs_dim, env.n_actions, env.n_agents, cfg)
    print(f"Device: {cfg.device}")
    print(f"Map: {env.rows}x{env.cols}, agents={env.n_agents}, exits={len(env.exits)}, actions={env.n_actions}")
    print(f"Save dir: {save_dir}")

    obs, info = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0
    episode_logs = []
    best_evacuated = -1
    best_steps = args.max_steps + 1

    num_updates = max(1, cfg.total_timesteps // cfg.rollout_steps)
    global_step = 0
    last_done_tensor = torch.zeros(env.n_agents, dtype=torch.float32, device=cfg.device)
    for update in range(1, num_updates + 1):
        buffer = RolloutBuffer(cfg.rollout_steps, env.n_agents, env.obs_dim, agent.global_obs_dim, cfg.device)
        for _ in range(cfg.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
            global_obs = obs_tensor.flatten()
            with torch.no_grad():
                actions, logprobs, values = agent.act(obs_tensor, global_obs)

            next_obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
            done = terminated or truncated
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
            done_tensor = torch.full((env.n_agents,), float(done), dtype=torch.float32, device=cfg.device)
            last_done_tensor = done_tensor
            buffer.add(obs_tensor, global_obs, actions, logprobs, reward_tensor, done_tensor, values)

            episode_reward += float(np.sum(rewards))
            episode_steps += 1
            global_step += 1
            obs = next_obs

            if done:
                num_evacuated = int(info["num_evacuated"])
                episode_logs.append(
                    {
                        "episode": episode_idx,
                        "global_step": global_step,
                        "episode_reward": round(episode_reward, 6),
                        "episode_steps": episode_steps,
                        "num_evacuated": num_evacuated,
                    }
                )
                is_better = num_evacuated > best_evacuated or (
                    num_evacuated == best_evacuated and episode_steps < best_steps
                )
                if is_better:
                    best_evacuated = num_evacuated
                    best_steps = episode_steps
                    plot_paths(env.grid_map, info["paths"], env.exits, save_dir / "best_paths.png")
                    agent.save(str(save_dir / "best_model.pt"))

                episode_idx += 1
                episode_reward = 0.0
                episode_steps = 0
                obs, info = env.reset()

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
        global_obs = obs_tensor.flatten()
        with torch.no_grad():
            next_value = agent.critic(global_obs.unsqueeze(0)).squeeze(0)
        next_done = last_done_tensor
        buffer.compute_returns_and_advantages(next_value, next_done, cfg.gamma, cfg.gae_lambda)
        metrics = agent.update(buffer)

        if update % 5 == 0 or update == 1:
            recent = episode_logs[-10:]
            mean_reward = np.mean([row["episode_reward"] for row in recent]) if recent else 0.0
            mean_evac = np.mean([row["num_evacuated"] for row in recent]) if recent else 0.0
            print(
                f"update={update:04d}/{num_updates} step={global_step:07d} "
                f"loss={metrics['loss']:.4f} entropy={metrics['entropy']:.4f} "
                f"recent_reward={mean_reward:.2f} recent_evacuated={mean_evac:.2f}"
            )

    agent.save(str(save_dir / "final_model.pt"))
    write_csv(save_dir / "training_log.csv", episode_logs)
    if episode_logs:
        plot_training_curves(save_dir / "training_log.csv", save_dir / "training_curves.png")
    print("Training finished.")


if __name__ == "__main__":
    main()
