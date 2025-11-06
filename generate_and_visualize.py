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

    # create subplot: top row = 3D scene + table, bottom row = time-series (spans both cols)
    fig = make_subplots(rows=2, cols=2,
               specs=[[{"type": "scene"}, {"type": "domain"}],
                   [{"colspan": 2, "type": "xy"}, None]],
               column_widths=[0.72, 0.28],
               row_heights=[0.65, 0.35],
               vertical_spacing=0.08)

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

    # prepare mean distance time-series (bottom subplot) and initial marker
    mean_dist_series = [m["mean_dist"] for m in timestep_metrics]
    # add static mean line (row=2, col=1 spanning both columns)
    fig.add_trace(go.Scatter(x=list(range(T)), y=mean_dist_series, mode="lines", name="mean_dist", line=dict(color="#444444")), row=2, col=1)
    # add moving marker (will be updated per-frame)
    fig.add_trace(go.Scatter(x=[0], y=[mean_dist_series[0]], mode="markers", marker=dict(size=10, color="red"), name="current_t"), row=2, col=1)

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

        # update marker for mean_dist
        marker = go.Scatter(x=[t], y=[mean_dist_series[t]], mode="markers", marker=dict(size=10, color="red"))

        # frame data order must match figure.data ordering:
        # [agent0..agentN] (row1,col1 traces), table (row1,col2), mean_line (row2,col1), marker (row2,col1)
        frame_data = frame_traces + [table, go.Scatter(x=list(range(T)), y=mean_dist_series, mode="lines", line=dict(color="#444444")), marker]
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



def enrich_with_ppo_metrics(ep_df, agent_df, positions, goals, goal_radius=0.5):
    """Add PPO-style aggregate metrics to ep_df and agent_df.

    - team_reward: sum of agent returns (already in sum_cumulative_reward)
    - team_success: whether all agents succeeded in the episode
    - cooperation_index: mean off-diagonal correlation between agents' episodic returns
    - sample_efficiency series: mean episodic reward per episode (returned separately)
    - convergence_stability: rolling variance of mean episodic reward
    """
    # per-agent episodic returns from agent_df
    per_episode = {}
    for _, row in agent_df.iterrows():
        ep = int(row["episode"])
        per_episode.setdefault(ep, []).append(row["cum_reward"])

    episodes = sorted(per_episode.keys())
    mean_rewards = []
    team_success = []
    coop_indices = []

    for ep in episodes:
        arr = np.array(per_episode[ep])
        mean_rewards.append(float(arr.mean()))
        # team success: all agents success in this episode
        ep_agent_rows = agent_df[agent_df["episode"] == ep]
        all_success = bool(ep_agent_rows["success"].all())
        team_success.append(int(all_success))

        # cooperation index: proxy via correlation of agent returns with team mean
        if arr.size > 1 and np.std(arr) > 0:
            team_mean = arr.mean()
            # correlation of returns with team mean; if degenerate set 0
            try:
                corr = float(np.corrcoef(arr, np.repeat(team_mean, arr.size))[0, 1])
            except Exception:
                corr = 0.0
            if np.isnan(corr):
                corr = 0.0
            coop_indices.append(corr)
        else:
            coop_indices.append(0.0)

    # attach new columns to ep_df by episode order
    ep_df = ep_df.copy()
    ep_df["team_success"] = [team_success[i] if i < len(team_success) else 0 for i in range(len(ep_df))]
    ep_df["cooperation_index"] = [coop_indices[i] if i < len(coop_indices) else 0.0 for i in range(len(ep_df))]
    ep_df["mean_episode_reward"] = [mean_rewards[i] if i < len(mean_rewards) else 0.0 for i in range(len(ep_df))]

    # sample efficiency timeseries (episode index -> mean reward)
    sample_efficiency = pd.DataFrame({"episode": episodes, "mean_episode_reward": mean_rewards})
    sample_efficiency["rolling_var_3"] = sample_efficiency["mean_episode_reward"].rolling(window=3, min_periods=1).var()

    return ep_df, agent_df, sample_efficiency


def plot_metrics_dashboard(ep_df, sample_efficiency, out_html="metrics_dashboard.html"):
    """Create a simple Plotly dashboard with mean episodic reward and team reward per episode."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Build a 3-row subplot: mean episodic reward, team reward, combined metrics
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Mean Episodic Reward", "Team Reward per Episode", "Collisions / Entropy / Advantage"))

    # Row 1: mean episodic reward (sample_efficiency)
    fig.add_trace(go.Scatter(x=sample_efficiency['episode'], y=sample_efficiency['mean_episode_reward'], mode='lines+markers', name='mean_episode_reward'), row=1, col=1)
    fig.update_yaxes(title_text='mean reward', row=1, col=1)

    # Row 2: team reward
    if 'sum_cumulative_reward' in ep_df.columns:
        fig.add_trace(go.Bar(x=ep_df['episode'], y=ep_df['sum_cumulative_reward'], name='team_reward'), row=2, col=1)
        fig.update_yaxes(title_text='team reward', row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=ep_df['episode'], y=ep_df['mean_cumulative_reward'], name='mean_agent_reward'), row=2, col=1)
        fig.update_yaxes(title_text='mean agent reward', row=2, col=1)

    # Row 3: collisions, mean_policy_entropy, mean_advantage if present
    if 'collisions' in ep_df.columns:
        fig.add_trace(go.Scatter(x=ep_df['episode'], y=ep_df['collisions'], mode='lines+markers', name='collisions'), row=3, col=1)
    if 'mean_policy_entropy' in ep_df.columns:
        fig.add_trace(go.Scatter(x=ep_df['episode'], y=ep_df['mean_policy_entropy'], mode='lines+markers', name='mean_policy_entropy'), row=3, col=1)
    if 'mean_advantage' in ep_df.columns:
        fig.add_trace(go.Scatter(x=ep_df['episode'], y=ep_df['mean_advantage'], mode='lines+markers', name='mean_advantage'), row=3, col=1)
    fig.update_yaxes(title_text='value', row=3, col=1)

    fig.update_xaxes(title_text='episode', row=3, col=1)

    # write to HTML
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved metrics dashboard to {out_html}")
    import plotly.express as px
    # line plot: mean episode reward
    fig1 = px.line(sample_efficiency, x="episode", y="mean_episode_reward", title="Mean Episodic Reward (per episode)")
    # bar plot: team reward from ep_df
    if "sum_cumulative_reward" in ep_df.columns:
        fig2 = px.bar(ep_df, x="episode", y="sum_cumulative_reward", title="Team Reward per Episode")
    else:
        fig2 = px.bar(ep_df, x="episode", y="mean_cumulative_reward", title="Mean Agent Reward per Episode")

    # combine into a single HTML file by writing separate divs
    html = "<html><head><meta charset=\"utf-8\"></head><body>"
    html += fig1.to_html(full_html=False, include_plotlyjs='cdn')
    html += "<hr>"
    html += fig2.to_html(full_html=False, include_plotlyjs=False)
    html += "</body></html>"

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved metrics dashboard to {out_html}")


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
    # Save baseline metrics
    save_metrics(ep_df, agent_df, out_dir="metrics_output")

    # Enrich with PPO-style aggregates (team success, cooperation index, sample-efficiency)
    ep_df2, agent_df2, sample_eff = enrich_with_ppo_metrics(ep_df, agent_df, positions, goals, goal_radius=0.5)
    # Save enriched metrics and dashboard
    save_metrics(ep_df2, agent_df2, out_dir="metrics_output")
    plot_metrics_dashboard(ep_df2, sample_eff, out_html="metrics_dashboard.html")

    # Create interactive episode visualization (per-agent table + 3D)
    visualize_episode_with_metrics(positions, goals, episode=0, output_html="episode0.html")
