# Learning RL by building it: from REINFORCE to PPO

I've been working on post-training LLMs with RL for a while now, and at some point I realized that I could read a GRPO paper and follow it fine, but if you asked me to sit down and *write* a policy gradient from scratch, I'd have to think really hard. Most of what I knew came from skimming code in big libraries where everything is abstracted three layers deep.

So I decided to just build it with PyTorch :D using a simple env like CartPole.
Basically start with REINFORCE, see where it goes wrong, and then try to build PPO on top of it.

## REINFORCE

So, the core idea of REINFORCE is super simple actually and very intuitive: **if an episode went well, make those actions more probable. If it went bad, make them less probable. And then scale by how good or bad it was.**

That's it. That's the whole algorithm xd

The math just makes that precise. Basically you want to maximize expected reward $$J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]$$, you can't differentiate through the sampling, but the log-derivative trick gives you something you can:

$$\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta} \left[ \sum\_{t} \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \cdot G\_t \right]$$

where $$G\_t$$ is the reward-to-go from timestep $$t$$. The cool thing about this is that the environment dynamics gets out of the gradient calculation, so you only need the log-probs of your own actions and that's it.  You also subtract the mean return as a baseline to reduce variance and you're done.

In code, the loss is really intuitive too:

```python
def compute_loss(all_log_probs, all_returns):
    trajectory_losses = []
    for log_probs, returns in zip(all_log_probs, all_returns):
        log_probs = torch.stack(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = returns - returns.mean()
        trajectory_losses.append(torch.sum(log_probs * advantage))
    return -torch.stack(trajectory_losses).mean()
```

The negative sign is there because PyTorch does gradient descent and we want ascent. One subtlety I tripped on: each `dist.log_prob(action)` is a tensor with a gradient graph attached to your network. If you do `torch.tensor(log_probs)` instead of `torch.stack(log_probs)`, you silently strip the graph and your gradients become zero. Took me longer than I want to admit to figure that out.

I plugged this into CartPole-v1 (4-d state, 2 actions, +1 reward per timestep up to 500), and it worked, kinda xd

## A tale of two learning rates

First run, `lr = 1e-2`:

```
  Iter    0  |  Avg Reward:   16.9
  Iter   50  |  Avg Reward:  373.7
  Iter  200  |  Avg Reward:  500.0   ← perfect!
  Iter  250  |  Avg Reward:  500.0   ← still perfect
  Iter  300  |  Avg Reward:  403.6   ← uh oh
  Iter  350  |  Avg Reward:  118.3   ← bruh 💀
  Iter  500  |  Avg Reward:  117.0   ← bruh x2 💀
```

![High learning rate reward curve](/assets/images/reward_curve_high_lr.png)

Look at that curve. It hits the maximum reward of 500 around iteration 100, holds it for a bit, and then just... falls off a cliff. And it never really recovers. This is REINFORCE's high-variance problem in one image: with only 10 trajectories per update and a high learning rate, one unlucky batch of episodes can produce a gradient big enough to wreck a perfectly good policy. There's nothing in the algorithm stopping that from happening.

Second run, `lr = 5e-4`:

![Low learning rate reward curve](/assets/images/reward_curve_lower_lr.png)

Way better. Slow but steady, monotonically improving, no collapses. By iteration 500 it's basically solved.

Two runs, same algorithm, totally different stories. And looking at this side by side made the weaknesses of vanilla REINFORCE click for me in a way no textbook had:

1. **High variance.** Your gradient is estimated from like 10 episodes. The returns can be huge. The estimates fluctuate wildly. With a high learning rate, you eventually take one bad step and blow everything up.
2. **Sample inefficiency.** You collect trajectories, do *one* gradient update, and throw them away. Because the moment you update $$\theta$$, those old trajectories are no longer from the current policy and you can't use them anymore. For CartPole this is fine. For a 7B language model where each "trajectory" is a generated response? Idk xd
3. **No update constraints.** Nothing prevents a single gradient step from being catastrophic. Your only protection is the learning rate, and as we just saw, tuning that is its own headache.

Each of these pain points has a name when you fix it. Together, the fixes are called PPO.

## PPO: REINFORCE with seatbelts

So, we can that the big idea of PPO is **importance sampling**. Instead of throwing trajectories away after one update, we reweight them. If you collected data under policy $$\pi\_{\theta\_{old}}$$ but now you're at $$\pi\_\theta$$, you can still use that data, just multiply by the ratio (this is what the whole importance sampling is about :D):

$$r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{old}}(a\_t|s\_t)}$$

This lets you do multiple gradient steps on the same batch (sample efficiency, fixed). But it opens a new problem: if $$\pi\_\theta$$ drifts too far from $$\pi\_{\theta\_{old}}$$, that ratio explodes and your gradient estimates become garbage. PPO's clipping is the seatbelt:

$$L^{CLIP}(\theta) = \mathbb{E}\_t \left[ \min\left( r\_t(\theta) A\_t, \text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) A\_t \right) \right]$$

If the ratio strays outside $$[1-\epsilon, 1+\epsilon]$$ in a direction that would help the policy, the gradient gets zeroed out. You literally cannot take a step bigger than $$\epsilon$$. (Catastrophic updates, fixed.)

The other piece is replacing the mean-return baseline with a learned value function $$V\_\phi(s)$$. This gives you a much better baseline, and combined with bootstrapping ([GAE](https://arxiv.org/abs/1506.02438)), you get lower variance advantage estimates:

$$\delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t)$$

$$A\_t = \delta\_t + \gamma \lambda A\_{t+1}$$

(High variance, fixed.)

In code, the heart of PPO is just this:

```python
ratio = torch.exp(log_prob_new - old_log_probs[t])
clipped_ratio = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
policy_loss -= torch.min(ratio * advantages[t], clipped_ratio * advantages[t])

value_target = advantages[t] + old_values[t]
value_loss += (value_new - value_target.detach()) ** 2
```

That's the whole algorithmic difference. The clip, the value head, the multiple sub-epochs over the same batch. Pretty cool ngl :))

Pd: The full code is up on GitHub if you want to play with it: [reinforce.py](https://github.com/eltonjohnfanboy/PGtoReasoning/blob/main/REINFORCE/reinforce.py) and [ppo.py](https://github.com/eltonjohnfanboy/PGtoReasoning/blob/main/PPO/ppo.py). 
