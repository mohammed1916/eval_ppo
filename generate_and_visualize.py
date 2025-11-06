import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os


def generate_trajectories(num_episodes=50,
                          num_agents=4,
                          T=200,
                          dt=0.1,
                          arena_size=10.0,
                          max_speed=1.0,
                          goal_radius=0.5,
                          seed=None):
    """
    Generate synthetic multi-agent 3D trajectories.
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
        # random starts and goals inside cube [-arena_size, arena_size]^3
        starts = rng.uniform(-arena_size, arena_size, size=(num_agents, 3))
        g = rng.uniform(-arena_size, arena_size, size=(num_agents, 3))
        goals[ep] = g
        pos = starts.copy()
        vel = np.zeros_like(pos)

        for t in range(T):
            # steering toward goal
            desired = g - pos
            dist = np.linalg.norm(desired, axis=1, keepdims=True) + 1e-8
            desired_dir = desired / dist
            # base speed proportional to remaining distance (clipped)
            base_speed = np.clip(dist.squeeze() / (T*dt) * 2.0, 0.0, max_speed)
            base_vel = (desired_dir * base_speed[:, None])
            # add Gaussian noise to simulate stochastic policy (PPO-like)
            noise = rng.normal(scale=0.3, size=pos.shape)
            # smooth the noise with small inertia to mimic policy smoothing
            action = 0.85 * vel + 0.15 * (base_vel + 0.4 * noise)
            speed = np.linalg.norm(action, axis=1, keepdims=True)
            # clip speed
            speed_factor = np.clip(speed, 1e-8, max_speed) / (speed + 1e-8)
            action = action * speed_factor
            vel = action
            pos = pos + vel * dt

            positions[ep, :, t, :] = pos
            velocities[ep, :, t, :] = vel

    return positions, velocities, goals


def compute_metrics(positions, velocities, goals, collision_distance=0.5, goal_radius=0.5):
    """
    Compute per-episode and per-agent metrics:
      - cumulative_reward (synthetic): negative distance-to-go summed
      - success (final distance < goal_radius)
      - collisions: count of time-steps where any pair is closer than collision_distance
      - smoothness: mean acceleration magnitude
      - energy: sum squared control (velocity changes)
    Returns a pandas.DataFrame with one row per episode and aggregated metrics, and per-agent detail as DataFrame.
    """
    num_episodes, num_agents, T, _ = positions.shape
    ep_rows = []
    agent_rows = []

    for ep in range(num_episodes):
        pos = positions[ep]            # (num_agents, T, 3)
        vel = velocities[ep]          # (num_agents, T, 3)
        goal = goals[ep]              # (num_agents, 3)

        # distances over time to each goal
        dists = np.linalg.norm(pos - goal[:, None, :], axis=2)  # (num_agents, T)
        # synthetic "reward": negative distance at each step, sum over time
        cumulative_rewards = -np.sum(dists, axis=1)  # per-agent
        successes = (dists[:, -1] <= goal_radius)            # per-agent success flag

        # collisions: at each time, check pairwise distances
        collisions_t = 0
        for t in range(T):
            p = pos[:, t, :]  # (num_agents,3)
            diffs = p[:, None, :] - p[None, :, :]
            pair_dists = np.linalg.norm(diffs, axis=-1)
            mask = np.triu(pair_dists < collision_distance, k=1)
            collisions_t += int(mask.sum())

        # smoothness: acceleration norm mean
        if vel.shape[1] > 1:
            acc = np.diff(vel, axis=1)  # approx accel per time index
            smoothness = float(np.mean(np.linalg.norm(acc, axis=-1)))  # averaged over agents & time
            energy = float(np.sum((np.diff(vel, axis=1) ** 2)))
        else:
            smoothness = 0.0
            energy = 0.0

        ep_row = {
            "episode": int(ep),
            "num_agents": int(num_agents),
            "mean_cumulative_reward": float(np.mean(cumulative_rewards)),
            "sum_cumulative_reward": float(np.sum(cumulative_rewards)),
            "mean_success_rate": float(np.mean(successes)),
            "collisions": int(collisions_t),
            "smoothness": float(smoothness),
            "energy": float(energy),
        }
        ep_rows.append(ep_row)

        for a in range(num_agents):
            if vel.shape[1] > 1:
                agent_energy = float(np.sum((np.diff(vel[a], axis=0) ** 2)))
            else:
                agent_energy = 0.0
            agent_rows.append({
                "episode": int(ep),
                "agent": int(a),
                "cum_reward": float(cumulative_rewards[a]),
                "success": bool(successes[a]),
                "agent_energy": agent_energy,
                "final_dist": float(dists[a, -1]),
            })

    ep_df = pd.DataFrame(ep_rows)
    agent_df = pd.DataFrame(agent_rows)
    return ep_df, agent_df


def visualize_episode(positions, episode=0, title="Multi-agent 3D Trajectories", output_html=None):
    """
    Visualize a single episode as an interactive Plotly 3D animation.
    positions: (num_episodes, num_agents, T, 3)
    """
    num_episodes, num_agents, T, _ = positions.shape
    pos = positions[episode]  # (num_agents, T, 3)
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    colors = colors * ((num_agents // len(colors)) + 1)

    frames = []
    for t in range(T):
        data = []
        for a in range(num_agents):
            xs = pos[a, :t+1, 0]
            ys = pos[a, :t+1, 1]
            zs = pos[a, :t+1, 2]
            data.append(go.Scatter3d(x=xs, y=ys, z=zs,
                                     mode="lines+markers",
                                     marker=dict(size=4),
                                     line=dict(width=3),
                                     name=f"agent-{a}",
                                     marker_symbol="circle",
                                     hoverinfo="name",
                                     showlegend=(t == 0),
                                     marker_color=colors[a]))
        frames.append(go.Frame(data=data, name=str(t)))

    init_data = []
    for a in range(num_agents):
        init_data.append(go.Scatter3d(
            x=pos[a, :1, 0], y=pos[a, :1, 1], z=pos[a, :1, 2],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=3),
            name=f"agent-{a}",
            marker_color=colors[a],
            showlegend=True
        ))

    layout = go.Layout(
        title=title,
        scene=dict(aspectmode="auto"),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                pad={"r": 10, "t": 10},
            )
        ],
        sliders=[{
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": str(k), "method": "animate"} for k in range(T)
            ]
        }]
    )

    fig = go.Figure(data=init_data, frames=frames, layout=layout)
    if output_html:
        fig.write_html(output_html)
        print(f"Saved interactive visualization to {output_html}")
    else:
        fig.show()


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
    # Basic example pipeline
    positions, velocities, goals = generate_trajectories(
        num_episodes=6,
        num_agents=4,
        T=100,
        dt=0.1,
        arena_size=8.0,
        max_speed=1.2,
        seed=42
    )

    ep_df, agent_df = compute_metrics(positions, velocities, goals, collision_distance=0.6)
    print(ep_df.head())
    save_metrics(ep_df, agent_df, out_dir="metrics_output")

    # Visualize episode 0 and save as HTML (open in browser)
    visualize_episode(positions, episode=0, title="Episode 0 - Multi-agent trajectories", output_html="episode0.html")
