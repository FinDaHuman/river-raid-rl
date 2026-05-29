import numpy as np

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.utils.logger import Logger


def train(agent, config):
    logger = Logger("logs", config.training.run_name)
    env = make_riverraid_env(config.env)

    state, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_num = 0
    best_mean_reward = float("-inf")
    eval_results = []

    for step in range(config.training.total_timesteps):
        action = agent.act(np.array(state), training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.memory.push(state, action, reward, next_state, terminated or truncated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if step >= agent.config.min_replay_size and step % agent.config.train_freq == 0:
            metrics = agent.update()
            if metrics and step % config.training.log_freq == 0:
                logger.log(step, metrics)

        if terminated or truncated:
            state, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_num += 1

        if step > 0 and step % config.training.eval_freq == 0:
            eval_result = evaluate(agent, config.env, config.training.eval_episodes)
            eval_results.append(eval_result)
            mean = eval_result["mean_reward"]
            logger.log(step, {
                "eval/mean_reward": mean,
                "eval/std_reward": eval_result["std_reward"],
                "eval/mean_length": eval_result["mean_length"],
            })
            if mean > best_mean_reward:
                best_mean_reward = mean
                agent.save(f"checkpoints/{config.training.run_name}/best.pt")
                logger.log(step, {"best_reward": best_mean_reward})

        if step > 0 and step % config.training.save_freq == 0:
            agent.save(f"checkpoints/{config.training.run_name}/step_{step}.pt")

    env.close()
    agent.save(f"checkpoints/{config.training.run_name}/final.pt")
    logger.save()
    return eval_results
