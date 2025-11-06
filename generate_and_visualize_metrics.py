import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def generate_trajectories(num_episodes=8,
                          num_agents=4,
                          T=120,
                          dt=0.1,
                          arena_size=8.0,
                          max_speed=1.0,
                          seed=None):
    """Generate synthetic multi-agent 3D trajectories.

    Returns:
      positions: np.array shape (num_episodes, num_agents, T, 3)
      velocities: np.array shape (num_episodes, num_agents, T, 3)
      goals: np.array shape (num_episodes, num_agents, 3)
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_episodes, num_agents, T, 3), dtype=float)
    velocities = np.zeros_like(positions)
    goals = np.zeros((num_episodes, num_agents, 3), dtype=float)

    for ep in range(num_episodes):
        starts = rng.uniform(-arena_size, arena_size, size=(num_agents, 3))
        g = rng.uniform(-arena_size, arena_size, size=(num_agents, 3))
        goals[ep] = g
        pos = starts.copy()
        vel = np.zeros_like(pos)

        for t in range(T):
            desired = g - pos
            dist = np.linalg.norm(desired, axis=1, keepdims=True) + 1e-8
            desired_dir = desired / dist
            base_speed = np.clip(dist.squeeze() / (T * dt) * 0.5, 0.0, max_speed)
            base_vel = desired_dir * base_speed[:, None]
            noise = rng.normal(scale=0.35, size=pos.shape)
            action = 0.8 * vel + 0.2 * (base_vel + 0.4 * noise)
            speed = np.linalg.norm(action, axis=1, keepdims=True)
            speed_factor = np.clip(speed, 1e-8, max_speed) / (speed + 1e-8)
            action = action * speed_factor
            vel = action
            pos = pos + vel * dt

            positions[ep, :, t, :] = pos
            velocities[ep, :, t, :] = vel

    return positions, velocities, goals


def compute_episode_metrics(positions, velocities, goals, collision_distance=0.5):
    """Compute summary metrics per episode and per agent."""
    num_episodes, num_agents, T, _ = positions.shape
    ep_rows = []
    agent_rows = []

    for ep in range(num_episodes):
        pos = positions[ep]
        vel = velocities[ep]
        goal = goals[ep]

        dists = np.linalg.norm(pos - goal[:, None, :], axis=2)
        cumulative_rewards = -np.sum(dists, axis=1)
        successes = (dists[:, -1] <= 0.5)

        collisions_t = 0
        for t in range(T):
            p = pos[:, t, :]
            diffs = p[:, None, :] - p[None, :, :]
            pair_dists = np.linalg.norm(diffs, axis=-1)
            mask = np.triu(pair_dists < collision_distance, k=1)
            collisions_t += int(mask.sum())

        acc = np.diff(vel, axis=1)
        smoothness = float(np.mean(np.linalg.norm(acc, axis=-1)))
        energy = float(np.sum((np.diff(vel, axis=1) ** 2)))

        # synthetic PPO metrics derived from distances over time
        # per-agent entropy: 0.3 + 0.7 * d/(1+d) averaged over time
        per_agent_entropy_time = 0.3 + 0.7 * (dists / (1.0 + dists))
        avg_entropy_per_agent = np.mean(per_agent_entropy_time, axis=1)

        # per-timestep normalization for advantage (higher when closer)
        maxd_per_t = np.max(dists, axis=0) + 1e-8
        per_t_adv = (maxd_per_t[None, :] - dists) / maxd_per_t[None, :]
        avg_adv_per_agent = np.mean(per_t_adv, axis=1)

        # synthetic KL estimate (small) derived from inter-agent variability
        kl_estimate = float(np.mean(np.std(dists, axis=1)) * 0.01)

        ep_rows.append({
            "episode": int(ep),
            "num_agents": int(num_agents),
            "mean_cumulative_reward": float(np.mean(cumulative_rewards)),
            "sum_cumulative_reward": float(np.sum(cumulative_rewards)),
            "mean_success_rate": float(np.mean(successes)),
            "collisions": int(collisions_t),
            "smoothness": smoothness,
            "energy": energy,
            "mean_policy_entropy": float(np.mean(avg_entropy_per_agent)),
            "mean_advantage": float(np.mean(avg_adv_per_agent)),
            "kl_estimate": kl_estimate,
        })

        for a in range(num_agents):
            agent_rows.append({
                "episode": int(ep),
                "agent": int(a),
                "cum_reward": float(cumulative_rewards[a]),
                "success": bool(successes[a]),
                "agent_energy": float(np.sum((np.diff(vel[a], axis=0) ** 2))) if vel.shape[1] > 1 else 0.0,
                "final_dist": float(dists[a, -1]),
                "avg_policy_entropy": float(avg_entropy_per_agent[a]),
                "avg_advantage": float(avg_adv_per_agent[a]),
            })

    return pd.DataFrame(ep_rows), pd.DataFrame(agent_rows)


def compute_timestep_metrics(pos, goals, collision_distance=0.5):
    """Compute metrics for each time-step in a single episode.

    pos: (num_agents, T, 3)
    goals: (num_agents, 3)
    Returns list of dicts length T with summary metrics for that t.
    """
    num_agents, T, _ = pos.shape
    metrics = []
    for t in range(T):
        p = pos[:, t, :]
        dists = np.linalg.norm(p - goals, axis=1)
        mean_dist = float(np.mean(dists))
        min_dist = float(np.min(dists))
        # collisions at this timestep
        diffs = p[:, None, :] - p[None, :, :]
        pair_dists = np.linalg.norm(diffs, axis=-1)
        collisions = int(np.triu(pair_dists < collision_distance, k=1).sum())

        # synthetic per-agent PPO-like metrics derived from distances
        # policy entropy: higher when agents are far from goal (more uncertainty)
        per_agent_entropy = [float(0.3 + 0.7 * (d / (1.0 + d))) for d in dists]
        # advantage (synthetic): normalized inverse distance (higher advantage when closer)
        maxd = float(np.max(dists) + 1e-8)
        per_agent_adv = [float((maxd - d) / (maxd)) for d in dists]

        metrics.append({
            "t": int(t),
            "mean_dist": mean_dist,
            "min_dist": min_dist,
            "collisions": collisions,
            "per_agent_dist": [float(x) for x in dists],
            "per_agent_entropy": per_agent_entropy,
            "per_agent_advantage": per_agent_adv,
        })
    return metrics


def visualize_episode_with_metrics(positions, goals, episode=0, output_html="episode0.html"):
    """Create a Plotly 3D animation with a per-agent metrics table that updates per-frame.

    The figure uses a 2-column layout: left is the 3D trajectories, right is a table
    showing per-agent distance-to-goal and instantaneous speed for the current frame.
    """
    num_episodes, num_agents, T, _ = positions.shape
    pos = positions[episode]
    timestep_metrics = compute_timestep_metrics(pos, goals[episode])

    # colors
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    colors = (base_colors * ((num_agents // len(base_colors)) + 1))[:num_agents]

    # create subplot: 3D scene + table
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "domain"}]],
                        column_widths=[0.72, 0.28])

    # initial 3D traces (one per agent)
    for a in range(num_agents):
        fig.add_trace(go.Scatter3d(
            x=pos[a, :1, 0], y=pos[a, :1, 1], z=pos[a, :1, 2],
            mode="lines+markers",
            marker=dict(size=4, color=colors[a]),
            line=dict(width=4, color=colors[a]),
            name=f"agent-{a}"
        ), row=1, col=1)

    # initial table (per-agent)
    # prepare initial per-agent metrics for t=0
    init_metrics = timestep_metrics[0]
    agent_names = [f"agent-{i}" for i in range(num_agents)]
    dists0 = [f"{v:.3f}" for v in init_metrics["per_agent_dist"]]
    speeds0 = ["0.000" for _ in range(num_agents)]
    entropy0 = [f"{v:.3f}" for v in init_metrics.get("per_agent_entropy", [0.0]*num_agents)]
    adv0 = [f"{v:.3f}" for v in init_metrics.get("per_agent_advantage", [0.0]*num_agents)]
    fig.add_trace(go.Table(
        header=dict(values=["agent", "dist_to_goal", "speed(frame)", "policy_entropy", "advantage"], align="left"),
        cells=dict(values=[agent_names, dists0, speeds0, entropy0, adv0], align="left")
    ), row=1, col=2)

    frames = []
    for t in range(T):
        frame_traces = []
        # 3D traces for each agent (path up to t)
        for a in range(num_agents):
            xs = pos[a, : t + 1, 0]
            ys = pos[a, : t + 1, 1]
            zs = pos[a, : t + 1, 2]
            frame_traces.append(go.Scatter3d(x=xs, y=ys, z=zs,
                                             mode="lines+markers",
                                             marker=dict(size=4, color=colors[a]),
                                             line=dict(width=4, color=colors[a]),
                                             name=f"agent-{a}"))

        # compute per-agent instantaneous speed (displacement per frame)
        speeds = []
        for a in range(num_agents):
            if t == 0:
                s = 0.0
            else:
                disp = pos[a, t, :] - pos[a, t - 1, :]
                s = float(np.linalg.norm(disp))
            speeds.append(f"{s:.3f}")

        m = timestep_metrics[t]
        dists = [f"{v:.3f}" for v in m["per_agent_dist"]]
        entropy = [f"{v:.3f}" for v in m.get("per_agent_entropy", [0.0]*num_agents)]
        adv = [f"{v:.3f}" for v in m.get("per_agent_advantage", [0.0]*num_agents)]

        # table trace with updated values
        table = go.Table(
            header=dict(values=["agent", "dist_to_goal", "speed(frame)", "policy_entropy", "advantage"], align="left"),
            cells=dict(values=[agent_names, dists, speeds, entropy, adv], align="left")
        )

        # frame data order must match figure.data ordering: first N agent 3D traces, then table
        frame_data = frame_traces + [table]
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # layout with animation controls
    fig.update_layout(title=f"Episode {episode} - multi-agent 3D trajectories",
                      scene=dict(aspectmode="auto"),
                      margin=dict(l=0, r=0, t=40, b=0),
                      updatemenus=[dict(type="buttons", showactive=False,
                                        y=0.05, x=0.1, xanchor="right", yanchor="top",
                                        pad=dict(t=45, r=10),
                                        buttons=[
                                            dict(label="Play", method="animate",
                                                 args=[None, dict(frame=dict(duration=60, redraw=True),
                                                                  transition=dict(duration=0),
                                                                  fromcurrent=True,
                                                                  mode="immediate")]),
                                            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                                        ])])

    # slider
    steps = []
    for k in range(T):
        steps.append({
            "method": "animate",
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(k)
        })
    sliders = [{"active": 0, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": steps}]
    fig.update_layout(sliders=sliders)

    # attach frames and write
    fig.frames = frames
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html)
    print(f"Saved interactive visualization to {output_html}")


def save_metrics(ep_df, agent_df, out_dir="metrics_output"):
    os.makedirs(out_dir, exist_ok=True)
    ep_csv = os.path.join(out_dir, "episode_metrics.csv")
    agent_csv = os.path.join(out_dir, "agent_metrics.csv")
    ep_df.to_csv(ep_csv, index=False)
    agent_df.to_csv(agent_csv, index=False)
    npz_path = os.path.join(out_dir, "metrics.npz")
    np.savez_compressed(npz_path, episode=ep_df.to_dict(orient="list"), agent=agent_df.to_dict(orient="list"))
    print(f"Saved metrics to {out_dir}")


if __name__ == "__main__":
    positions, velocities, goals = generate_trajectories(num_episodes=6, num_agents=5, T=150, dt=0.1, arena_size=6.0, max_speed=1.2, seed=123)
    ep_df, agent_df = compute_episode_metrics(positions, velocities, goals, collision_distance=0.6)
    print(ep_df.head())
    save_metrics(ep_df, agent_df, out_dir="metrics_output")
    visualize_episode_with_metrics(positions, goals, episode=0, output_html="episode0.html")
