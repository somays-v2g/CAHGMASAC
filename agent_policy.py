
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- Utils

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed=None):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)
        self.reset()
    def reset(self):
        self.s = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, 1), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.sn = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.full = False
    def push(self, s, a, r, sn, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.sn[self.ptr] = sn
        self.d[self.ptr] = d
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def push_batch(self, s, a, r, sn, d):
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
        r = np.asarray(r, dtype=np.float32).reshape(-1, 1)
        sn = np.asarray(sn, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32).reshape(-1, 1)
        n = int(len(s))
        if n == 0:
            return
        if not (len(a) == len(r) == len(sn) == len(d) == n):
            raise ValueError("ReplayBuffer.push_batch received mismatched batch lengths")

        if n >= self.capacity:
            s, a, r, sn, d = s[-self.capacity:], a[-self.capacity:], r[-self.capacity:], sn[-self.capacity:], d[-self.capacity:]
            self.s[:] = s
            self.a[:] = a
            self.r[:] = r
            self.sn[:] = sn
            self.d[:] = d
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.s[sl] = s
            self.a[sl] = a
            self.r[sl] = r
            self.sn[sl] = sn
            self.d[sl] = d
        else:
            first = self.capacity - self.ptr
            self.s[self.ptr:] = s[:first]
            self.a[self.ptr:] = a[:first]
            self.r[self.ptr:] = r[:first]
            self.sn[self.ptr:] = sn[:first]
            self.d[self.ptr:] = d[:first]
            rem = n - first
            self.s[:rem] = s[first:]
            self.a[:rem] = a[first:]
            self.r[:rem] = r[first:]
            self.sn[:rem] = sn[first:]
            self.d[:rem] = d[first:]

        if end >= self.capacity:
            self.full = True
        self.ptr = end % self.capacity
    def __len__(self):
        return self.capacity if self.full else self.ptr
    def sample(self, batch_size: int):
        n = len(self)
        if n == 0:
            return None
        idx = self.rng.choice(n, size=min(batch_size, n), replace=False)
        import torch
        return (
            torch.from_numpy(self.s[idx]).float(),
            torch.from_numpy(self.a[idx]).float(),
            torch.from_numpy(self.r[idx]).float(),
            torch.from_numpy(self.sn[idx]).float(),
            torch.from_numpy(self.d[idx]).float(),
        )


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()

        if isinstance(hidden, int):
            hidden_layers = ()
            feat_dim = hidden
        else:
            hidden = tuple(hidden)
            hidden_layers = hidden[:-1] if len(hidden) > 1 else ()
            feat_dim = hidden[-1] if len(hidden) >= 1 else 256 

        self.net = MLP(obs_dim, feat_dim, hidden=hidden_layers)
        self.mu = nn.Linear(feat_dim, act_dim)
        self.log_std = nn.Linear(feat_dim, act_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std) + 1e-6
        # 출력 NaN 방지
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
        std = torch.nan_to_num(std, nan=1.0, posinf=1e3, neginf=1e-6)
        return mu, std

    def sample(self, obs):
        mean, std = self(obs)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f"Actor produced non-finite params: mean={mean}, std={std}")
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)  # [-1,1]
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), mean
    
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=1, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden=hidden)

    def forward(self, obs):
        x = self.net(obs)
        return torch.tanh(x)

    def sample(self, obs, explore=False, noise_std=0.1):
        mean = self(obs)
        if explore:
            noise = noise_std * torch.randn_like(mean)
            action = torch.clamp(mean + noise, -1.0, 1.0)
        else:
            action = mean

        dummy_logp = torch.zeros(mean.size(0), 1, device=mean.device)
        return action, dummy_logp, mean



class QCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256,256)):
        super().__init__()
        self.net = MLP(obs_dim + 1, 1, hidden=hidden)
    def forward(self, obs, a):
        x = torch.cat([obs, a], dim=-1)
        return self.net(x)

# ---------- MultiAgent SAC with parameter sharing per role

class RoleSAC:
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        hs = cfg.get("hidden_sizes", [256,256])
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 2048)
        self.learning_starts = cfg.get("learning_starts", max(self.batch_size, 1000))
        
        self.buf_size = cfg.get("buffer_size", 200000)
        _sd = cfg.get("seed", None)
        self.buffer = ReplayBuffer(self.buf_size, obs_dim,
                                   seed=None if _sd is None else [_sd, 11, obs_dim])
        

        self.actor = GaussianPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)

        self.q1 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(
            np.log(0.2), requires_grad=True, device=self.device
        )

        target_entropy_coef = cfg.get("target_entropy_coef", 0.98)
        self.target_entropy = -target_entropy_coef  # for 1D action

        lr_a = cfg.get("actor_lr", 3e-4)
        lr_c = cfg.get("critic_lr", 3e-4)
        lr_alpha = cfg.get("alpha_lr", 3e-4)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=lr_c)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=lr_c)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs_vec: np.ndarray, explore: bool = True) -> float:
        obs_t = torch.from_numpy(obs_vec[None, :]).float().to(self.device)
        #obs_t = torch.from_numpy(obs_vec[None, :]).float()
        with torch.no_grad():
            if explore:
                a, _, _ = self.actor.sample(obs_t)
            else:
                _, _, mean = self.actor.sample(obs_t)
                a = mean
        return float(a.squeeze().cpu().numpy())

    def push(self, s, a, r, sn, d):
        self.buffer.push(s, a, r, sn, d)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def learn_step(self, demo_frac):
        
        if len(self.buffer) < self.learning_starts:
            return None
    
        batch = self.buffer.sample(self.batch_size)

        s, a, r, sn, d = batch
        if isinstance(s, np.ndarray):  
            s  = torch.from_numpy(s).float()
            a  = torch.from_numpy(a).float()
            r  = torch.from_numpy(r).float()
            sn = torch.from_numpy(sn).float()
            d  = torch.from_numpy(d).float()

        s, a, r, sn, d = [x.to(self.device) for x in (s, a, r, sn, d)]

        with torch.no_grad():
            an, logp_an, _ = self.actor.sample(sn)
            q1_t = self.q1_target(sn, an)
            q2_t = self.q2_target(sn, an)
            q_min = torch.min(q1_t, q2_t)
            y = r + (1 - d) * self.gamma * (q_min - self.alpha * logp_an)

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        loss_q1 = nn.MSELoss()(q1, y)
        loss_q2 = nn.MSELoss()(q2, y)
        self.opt_q1.zero_grad(); loss_q1.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); loss_q2.backward(); self.opt_q2.step()

        a_pi, logp_a, _ = self.actor.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        loss_actor = (self.alpha * logp_a - q_pi).mean()
        self.opt_actor.zero_grad(); loss_actor.backward(); self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp_a + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        metrics = {
            "critic_loss_q1": loss_q1.item(),
            "critic_loss_q2": loss_q2.item(),
            "td_error_q1": (q1 - y).abs().mean().item(),
            "td_error_q2": (q2 - y).abs().mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": y.mean().item(),
            "critic_loss": ((loss_q1 + loss_q2) / 2).item(),
            "actor_loss": loss_actor.item(),
            "td_error": (q1 - y).abs().mean().item(),
            "alpha": self.alpha.item(),
        }        
        return metrics

        
class RoleHSAC:
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        hs = cfg.get("hidden_sizes", [256,256])
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 2048)
        
        self.buf_size = cfg.get("buffer_size", 200000)
        _sd = cfg.get("seed", None)
        self.buffer_demo = ReplayBuffer(self.buf_size, obs_dim,
                                        seed=None if _sd is None else [_sd, 21, obs_dim])
        self.buffer_rl   = ReplayBuffer(self.buf_size, obs_dim,
                                        seed=None if _sd is None else [_sd, 22, obs_dim])

        self.learning_starts = 0
        

        self.actor = GaussianPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)

        self.q1 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(
            np.log(0.2), requires_grad=True, device=self.device
        )

        target_entropy_coef = cfg.get("target_entropy_coef", 0.98)
        self.target_entropy = -target_entropy_coef  # for 1D action

        lr_a = cfg.get("actor_lr", 3e-4)
        lr_c = cfg.get("critic_lr", 3e-4)
        lr_alpha = cfg.get("alpha_lr", 3e-4)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=lr_c)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=lr_c)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs_vec: np.ndarray, explore: bool = True) -> float:
        obs_t = torch.from_numpy(obs_vec[None, :]).float().to(self.device)
        with torch.no_grad():
            if explore:
                a, _, _ = self.actor.sample(obs_t)
            else:
                _, _, mean = self.actor.sample(obs_t)
                a = mean
        return float(a.squeeze().cpu().numpy())

    def push(self, s, a, r, sn, d):
        self.buffer.push(s, a, r, sn, d)
        
    def push_demo(self, s, a, r, sn, d):
        self.buffer_demo.push(s, a, r, sn, d)

    def push_rl(self, s, a, r, sn, d):
        self.buffer_rl.push(s, a, r, sn, d)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def learn_step(self, demo_frac):
        
        if len(self.buffer_rl) == 0 and len(self.buffer_demo) == 0:
            return
    
        batch_size = self.batch_size
        demo_bs = int(batch_size * demo_frac)
        rl_bs   = batch_size - demo_bs

        batch_rl   = self.buffer_rl.sample(rl_bs)   if rl_bs   > 0 else None
        batch_demo = self.buffer_demo.sample(demo_bs) if demo_bs > 0 else None

        if batch_rl is None and batch_demo is None:
            return

        def _to_device_batch(batch):
            if batch is None:
                return None
            out = []
            for x in batch:
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                out.append(x.to(self.device))
            return out

        batch_rl   = _to_device_batch(batch_rl)
        batch_demo = _to_device_batch(batch_demo)

        def _merge(b1, b2):
            if b1 is None:
                return b2
            if b2 is None:
                return b1
            return [torch.cat([x1, x2], dim=0) for x1, x2 in zip(b1, b2)]

        batch = _merge(batch_rl, batch_demo)
        if batch is None:
            return

        s, a, r, sn, d = batch  
        

        with torch.no_grad():
            an, logp_an, _ = self.actor.sample(sn)
            q1_t = self.q1_target(sn, an)
            q2_t = self.q2_target(sn, an)
            q_min = torch.min(q1_t, q2_t)
            y = r + (1 - d) * self.gamma * (q_min - self.alpha * logp_an)

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        loss_q1 = nn.MSELoss()(q1, y)
        loss_q2 = nn.MSELoss()(q2, y)
        self.opt_q1.zero_grad(); loss_q1.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); loss_q2.backward(); self.opt_q2.step()

        a_pi, logp_a, _ = self.actor.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        loss_actor = (self.alpha * logp_a - q_pi).mean()
        self.opt_actor.zero_grad(); loss_actor.backward(); self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp_a + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        metrics = {
            "critic_loss_q1": loss_q1.item(),
            "critic_loss_q2": loss_q2.item(),
            "td_error_q1": (q1 - y).abs().mean().item(),
            "td_error_q2": (q2 - y).abs().mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": y.mean().item(),
            "critic_loss": ((loss_q1 + loss_q2) / 2).item(),
            "actor_loss": loss_actor.item(),
            "td_error": (q1 - y).abs().mean().item(),
            "alpha": self.alpha.item(),
        }        
        return metrics


class RoleHTD3:
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hs = cfg.get("hidden_sizes", [256, 256])
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 1024)
        self.learning_starts = cfg.get("learning_starts", max(self.batch_size, 1000))
        
        self.buf_size = cfg.get("buffer_size", 200000)
        _sd = cfg.get("seed", None)
        self.buffer_demo = ReplayBuffer(self.buf_size, obs_dim,
                                        seed=None if _sd is None else [_sd, 31, obs_dim])
        self.buffer_rl   = ReplayBuffer(self.buf_size, obs_dim,
                                        seed=None if _sd is None else [_sd, 32, obs_dim])
    
        self.policy_noise = cfg.get("policy_noise", 0.2)
        self.noise_clip = cfg.get("noise_clip", 0.5)
        self.policy_delay = cfg.get("policy_delay", 2)
        self.total_it = 0

        self.diag_algo = str(cfg.get("algo", "td3")).upper()
        self.diag_actor_updates = 0
        self.diag_actor_limit = int(cfg.get("diag_actor_limit", 30))
        self.diag_actor_enabled = (obs_dim == 10 and self.diag_algo in ("TD3", "DTD3"))

        self.actor = DeterministicPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)
        self.actor_target = DeterministicPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        lr_a = cfg.get("actor_lr", 3e-4)
        lr_c = cfg.get("critic_lr", 3e-4)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=lr_c)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=lr_c)

    def act(self, obs_vec: np.ndarray, explore: bool = True, expl_noise: float = 0.02) -> float:
        obs_t = torch.from_numpy(obs_vec[None, :]).float().to(self.device)
        with torch.no_grad():
            a = self.actor(obs_t)
        a = a.cpu().numpy().flatten()
        if explore:
            a = a + np.random.normal(0, expl_noise, size=a.shape)
        a = np.clip(a, -1.0, 1.0)
        return float(a[0])

        
    def push_demo(self, s, a, r, sn, d):
        self.buffer_demo.push(s, a, r, sn, d)

    def push_rl(self, s, a, r, sn, d):
        self.buffer_rl.push(s, a, r, sn, d)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            
    def learn_step(self, demo_frac):
        if (len(self.buffer_rl) + len(self.buffer_demo)) < self.learning_starts:
            return None

        batch_size = self.batch_size
        demo_bs = int(batch_size * demo_frac)
        rl_bs   = batch_size - demo_bs

        batch_rl   = self.buffer_rl.sample(rl_bs) if rl_bs > 0 else None
        batch_demo = self.buffer_demo.sample(demo_bs) if demo_bs > 0 else None

        if batch_rl is None and batch_demo is None:
            return

        def _to_device_batch(batch):
            if batch is None:
                return None
            out = []
            for x in batch:
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                out.append(x.to(self.device))
            return out

        batch_rl   = _to_device_batch(batch_rl)
        batch_demo = _to_device_batch(batch_demo)

        def _merge(b1, b2):
            if b1 is None: return b2
            if b2 is None: return b1
            return [torch.cat([x1, x2], dim=0) for x1, x2 in zip(b1, b2)]

        batch = _merge(batch_rl, batch_demo)
        if batch is None:
            return

        s, a, r, sn, d = batch  # 모두 device tensor

        self.total_it += 1

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_next = (self.actor_target(sn) + noise).clamp(-1.0, 1.0)

            q1_t = self.q1_target(sn, a_next)
            q2_t = self.q2_target(sn, a_next)
            q_min = torch.min(q1_t, q2_t)

            y = r + (1.0 - d) * self.gamma * q_min

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        loss_q1 = nn.MSELoss()(q1, y)
        loss_q2 = nn.MSELoss()(q2, y)

        self.opt_q1.zero_grad()
        loss_q1.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        loss_q2.backward()
        self.opt_q2.step()

        if self.total_it % self.policy_delay == 0:
            a_pi = self.actor(s)
            actor_loss = -self.q1(s, a_pi).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)
            self.soft_update(self.actor, self.actor_target)
        
        metrics = {
            "critic_loss_q1": loss_q1.item(),
            "critic_loss_q2": loss_q2.item(),
            "td_error_q1": (q1 - y).abs().mean().item(),
            "td_error_q2": (q2 - y).abs().mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": y.mean().item(),
            "critic_loss": ((loss_q1 + loss_q2) / 2).item(),
            "actor_loss": actor_loss.item() if self.total_it % self.policy_delay == 0 else None,
            "td_error": (q1 - y).abs().mean().item(),
        }
        return metrics

class RoleTD3:
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hs = cfg.get("hidden_sizes", [256, 256])
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 1024)
        self.learning_starts = cfg.get("learning_starts", max(self.batch_size, 2000))

        self.buf_size = cfg.get("buffer_size", 200000)
        _sd = cfg.get("seed", None)
        self.buffer = ReplayBuffer(self.buf_size, obs_dim,
                                   seed=None if _sd is None else [_sd, 41, obs_dim])

        self.policy_noise = cfg.get("policy_noise", 0.2)
        self.noise_clip = cfg.get("noise_clip", 0.5)
        self.policy_delay = cfg.get("policy_delay", 2)
        self.total_it = 0

        self.diag_algo = str(cfg.get("algo", "td3")).upper()
        self.diag_actor_updates = 0
        self.diag_actor_limit = int(cfg.get("diag_actor_limit", 30))
        self.diag_actor_enabled = (obs_dim == 10 and self.diag_algo in ("TD3", "DTD3"))

        self.actor = DeterministicPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)
        self.actor_target = DeterministicPolicy(obs_dim, act_dim=1, hidden=hs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2 = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q2_target = QCritic(obs_dim, hidden=hs).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        lr_a = cfg.get("actor_lr", 3e-4)
        lr_c = cfg.get("critic_lr", 3e-4)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=lr_c)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=lr_c)

    
    def act(self, obs_vec: np.ndarray, explore: bool = True, expl_noise: float = 0.2) -> float:
        obs_t = torch.from_numpy(obs_vec[None, :]).float().to(self.device)
        with torch.no_grad():
            a = self.actor(obs_t)
        a = a.cpu().numpy().flatten()
        if explore:
            a = a + np.random.normal(0, expl_noise, size=a.shape)
        a = np.clip(a, -1.0, 1.0)
        return float(a[0])

    
    def push(self, s, a, r, sn, d):
        self.buffer.push(s, a, r, sn, d)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    
    def learn_step(self, demo_frac):
        if len(self.buffer) < self.learning_starts:
            return None

        batch = self.buffer.sample(self.batch_size)

        def _to_device(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            return x.to(self.device)

        s, a, r, sn, d = [_to_device(x) for x in batch]

        self.total_it += 1

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_next = (self.actor_target(sn) + noise).clamp(-1.0, 1.0)

            q1_t = self.q1_target(sn, a_next)
            q2_t = self.q2_target(sn, a_next)
            q_min = torch.min(q1_t, q2_t)

            y = r + (1.0 - d) * self.gamma * q_min

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        loss_q1 = nn.MSELoss()(q1, y)
        loss_q2 = nn.MSELoss()(q2, y)

        self.opt_q1.zero_grad()
        loss_q1.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        loss_q2.backward()
        self.opt_q2.step()

        if self.total_it % self.policy_delay == 0:
            a_pi = self.actor(s)
            actor_loss = -self.q1(s, a_pi).mean()

            do_diag = self.diag_actor_enabled and self.diag_actor_updates < self.diag_actor_limit
            if do_diag:
                with torch.no_grad():
                    pre_mean = a_pi.mean().item()
                    pre_std = a_pi.std(unbiased=False).item()
                    pre_min = a_pi.min().item()
                    pre_max = a_pi.max().item()
                    q_pi_before = self.q1(s, a_pi).mean().item()

                a_probe = a_pi.detach().clone().requires_grad_(True)
                q_probe = self.q1(s, a_probe).mean()
                dq_da = torch.autograd.grad(q_probe, a_probe, retain_graph=False, create_graph=False)[0]
                dqda_mean = dq_da.mean().item()
                dqda_abs = dq_da.abs().mean().item()
                dqda_min = dq_da.min().item()
                dqda_max = dq_da.max().item()

            self.opt_actor.zero_grad()
            actor_loss.backward()

            if do_diag:
                grad_sq = 0.0
                param_sq = 0.0
                grad_max = 0.0
                for p in self.actor.parameters():
                    param_sq += float(p.detach().pow(2).sum().item())
                    if p.grad is not None:
                        grad_sq += float(p.grad.detach().pow(2).sum().item())
                        grad_max = max(grad_max, float(p.grad.detach().abs().max().item()))
                grad_norm = grad_sq ** 0.5
                param_norm = param_sq ** 0.5

            self.opt_actor.step()

            if do_diag:
                with torch.no_grad():
                    a_after = self.actor(s)
                    post_mean = a_after.mean().item()
                    post_std = a_after.std(unbiased=False).item()
                    post_min = a_after.min().item()
                    post_max = a_after.max().item()
                    q_pi_after = self.q1(s, a_after).mean().item()
                self.diag_actor_updates += 1
                print(
                    f"[{self.diag_algo} ACTOR UPDATE {self.diag_actor_updates:03d}] "
                    f"it={self.total_it} | loss={actor_loss.item():+.6f} | "
                    f"before={pre_mean:+.4f}±{pre_std:.4f}[{pre_min:+.4f},{pre_max:+.4f}] -> "
                    f"after={post_mean:+.4f}±{post_std:.4f}[{post_min:+.4f},{post_max:+.4f}] | "
                    f"Qpi={q_pi_before:+.4f}->{q_pi_after:+.4f} | "
                    f"dQ/da mean={dqda_mean:+.3e} abs={dqda_abs:.3e} "
                    f"range=[{dqda_min:+.3e},{dqda_max:+.3e}] | "
                    f"gradNorm={grad_norm:.3e} gradMax={grad_max:.3e} paramNorm={param_norm:.3e}"
                )

            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)
            self.soft_update(self.actor, self.actor_target)
        
        metrics = {
            "critic_loss_q1": loss_q1.item(),
            "critic_loss_q2": loss_q2.item(),
            "td_error_q1": (q1 - y).abs().mean().item(),
            "td_error_q2": (q2 - y).abs().mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": y.mean().item(),
            "critic_loss": ((loss_q1 + loss_q2) / 2).item(),
            "actor_loss": actor_loss.item() if self.total_it % self.policy_delay == 0 else None,
            "td_error": (q1 - y).abs().mean().item(),
        }
        return metrics


class MultiAgentSAC:

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
        buffer_frac: float = 0.2,
    ):  
        
        if cfg is None:
            cfg = {}
            
        self.cfg = cfg
        self.steps = cfg.get("steps_hint", 96)  

        self.updates_per_step = cfg.get("updates_per_step", 1)
        self.demo_frac = buffer_frac
        self.n_updates = 0

        self.role_obs_dims = {
            "V2G": 10,
            "DERMS": 5
        }
        
        algo = cfg.get("algo", "sac")  

        if algo.lower() == "hsac":
            RoleAlgo = RoleHSAC
        elif algo.lower() == "sac":
            RoleAlgo = RoleSAC
        elif algo.lower() == "htd3":
            RoleAlgo = RoleHTD3
        elif algo.lower() == "td3":
            RoleAlgo = RoleTD3
        elif algo.lower() == "dsac":
            RoleAlgo = RoleSAC
        elif algo.lower() == "dtd3":
            RoleAlgo = RoleTD3
        else:
            raise ValueError(f"Unknown algo: {algo}")
        
        self.role_algo: Dict[str, RoleAlgo] = {
            role: RoleAlgo(self.role_obs_dims[role], cfg)
            for role in self.role_obs_dims
        }
        self._last_obs_by_agent = {}
        self._last_actions = {}
        self.actions = {}

        self.algo_name = str(algo).lower()
        self.diag_enabled = self.algo_name in ("td3", "dtd3")
        self._diag_step_count = 0
        self._diag_episode = 0
        self._diag_pending_print = False
        self._diag_raw_actions = []
        self._diag_env_actions = []
        self._diag_rewards = []
        self._diag_last_learn = {}
        self._diag_raw_last_by_agent = {}

    def role_of(self, aid: str) -> str:
        if aid.startswith("V2G") or aid.startswith("EV_"):
            return "V2G"
        if aid.startswith("DERMS"):
            return "DERMS"
        return None

    def obs_to_vec_per_role(self, role: str, obs: dict) -> np.ndarray:
  
        if role == "V2G":

            t_idx = float(obs.get("t_idx", 0.0))
            steps = float(self.steps or 1.0)
            t_norm = t_idx / steps

            is_peak = 1.0 if obs.get("is_peak", False) else 0.0
            conn = 1.0 if obs.get("connectivity", False) else 0.0
            vpp_req = 1.0 if obs.get("vpp_req", False) else 0.0

            ca = obs.get("current_act", "Idle")
            if ca == "Idle":
                current_act = 0.25
            elif ca == "Charging":
                current_act = 0.75
            elif ca == "V2H":
                current_act = -0.25
            elif ca == "V2G":
                current_act = -0.75  
            else:
                current_act = 0.0  # fallback

            current_soc = float(obs.get("current_soc", obs.get("soc_after_trip", 0.0)))

            soc_after_trip = obs['soc_after_trip']
            V2G_availability = obs['V2G_availability']
            V2H_power = obs["V2H_power"] / 7
            V2G_power = obs["V2G_power"] / 7
            price = obs["price"] / 55

            vec = np.array(
                [t_norm, is_peak, conn, current_act, current_soc, soc_after_trip, V2G_availability, V2H_power, V2G_power, price],
                dtype=np.float64,
            )

        elif role == "DERMS":

            t_idx = float(obs.get("t_idx", 0.0))
            steps = float(self.steps or 1.0)
            t_norm = t_idx / steps

            summary = obs.get("summary") or {}
            vpu_min = float(summary.get("vpu_min", 1.0))
            vpu_max = float(summary.get("vpu_max", 1.0))
            vpu_mean = float(summary.get("vpu_mean", 1.0))
            vpp_req = 1.0 if obs.get("VPP_req", False) else 0.0

            vec = np.array(
                [t_norm, vpu_min, vpu_max, vpu_mean, vpp_req],
                dtype=np.float64,
            )

        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        vec = np.clip(vec, -1e6, 1e6).astype(np.float32)
        return vec

    
    def act(self, obs_by_agent, explore: bool = True, expl_noise: float = 0.2):
        derms_ids = [aid for aid in obs_by_agent if self.role_of(aid) == "DERMS"]
        ev_ids = [aid for aid in obs_by_agent if self.role_of(aid) != "DERMS"]
        actions = {}

        if ev_ids:
            role = "V2G"
            obs_batch = np.stack(
                [self.obs_to_vec_per_role(role, obs_by_agent[aid]) for aid in ev_ids],
                axis=0,
            )
            role_algo = self.role_algo[role]
            obs_t = torch.from_numpy(obs_batch).float().to(role_algo.device)

            with torch.no_grad():
                if isinstance(role_algo.actor, GaussianPolicy):
                    if explore:
                        a, _, _ = role_algo.actor.sample(obs_t)
                    else:
                        mean, _ = role_algo.actor(obs_t)
                        a = torch.tanh(mean)
                else:
                    raw_a = role_algo.actor(obs_t)
                    a = raw_a
                    if explore:
                        a = a + torch.randn_like(a) * float(expl_noise)
                    a = torch.clamp(a, -1.0, 1.0)

                    if self.diag_enabled:
                        raw_np = raw_a.squeeze(-1).detach().cpu().numpy()
                        for _aid, _raw in zip(ev_ids, raw_np):
                            self._diag_raw_last_by_agent[_aid] = float(_raw)

            a = a.squeeze(-1).cpu().numpy()
            for aid, ai in zip(ev_ids, a):
                actions[aid] = float(np.clip(ai, -1.0, 1.0))
                self._last_obs_by_agent[aid] = obs_by_agent[aid]
                self._last_actions[aid] = actions[aid]

        for aid in derms_ids:
            role = "DERMS"
            obs_vec = self.obs_to_vec_per_role(role, obs_by_agent[aid])
            role_algo = self.role_algo[role]
            actions[aid] = role_algo.act(obs_vec, explore=explore)
            self._last_obs_by_agent[aid] = obs_by_agent[aid]
            self._last_actions[aid] = actions[aid]

        self.actions = actions
        return actions


    def observe(
        self,
        last_actions: Dict[str, dict],
        last_obs_by_agent: Dict[str, dict],
        next_obs_by_agent: Dict[str, dict],
        reward_by_agent: Dict[str, dict],
        policy: str,
        done: int,
    ):

        if self.diag_enabled and policy != "Heuristic":
            ev_aids = [aid for aid in last_obs_by_agent if self.role_of(aid) == "V2G"
                       and aid in reward_by_agent]
            if ev_aids:
                try:
                    obs_batch = np.stack(
                        [self.obs_to_vec_per_role("V2G", last_obs_by_agent[aid]) for aid in ev_aids],
                        axis=0,
                    )
                    role_algo = self.role_algo["V2G"]
                    obs_t = torch.from_numpy(obs_batch).float().to(role_algo.device)
                    with torch.no_grad():
                        raw_now = role_algo.actor(obs_t)
                        if isinstance(raw_now, tuple):
                            raw_now = torch.tanh(raw_now[0])
                    raw_np = raw_now.squeeze(-1).detach().cpu().numpy()
                except Exception:
                    raw_np = np.full(len(ev_aids), np.nan, dtype=float)

                for aid, raw_val in zip(ev_aids, raw_np):
                    self._diag_raw_actions.append(float(raw_val))
                    self._diag_env_actions.append(float(last_actions.get(aid, 0.0)))
                    self._diag_rewards.append(float(reward_by_agent[aid]))
            self._diag_step_count += 1
            if self._diag_step_count >= int(self.steps):
                self._diag_pending_print = True

        role_rows = {role: {"s": [], "a": [], "r": [], "sn": []} for role in self.role_algo}

        for aid, last_obs in last_obs_by_agent.items():
            role = self.role_of(aid)
            if role not in self.role_algo:
                continue
            if aid not in next_obs_by_agent or aid not in reward_by_agent:
                continue
            rows = role_rows[role]
            rows["s"].append(self.obs_to_vec_per_role(role, last_obs))
            rows["sn"].append(self.obs_to_vec_per_role(role, next_obs_by_agent[aid]))
            rows["a"].append(float(last_actions.get(aid, 0.0)))
            rows["r"].append(float(reward_by_agent[aid]))

        for role, rows in role_rows.items():
            if not rows["s"]:
                continue
            algo_agent = self.role_algo[role]
            s = np.asarray(rows["s"], dtype=np.float32)
            sn = np.asarray(rows["sn"], dtype=np.float32)
            a = np.asarray(rows["a"], dtype=np.float32).reshape(-1, 1)
            r = np.asarray(rows["r"], dtype=np.float32).reshape(-1, 1)
            d = np.full((len(s), 1), float(done), dtype=np.float32)

            if policy == "Heuristic" and hasattr(algo_agent, "buffer_demo"):
                target_buffer = algo_agent.buffer_demo
            elif policy != "Heuristic" and hasattr(algo_agent, "buffer_rl"):
                target_buffer = algo_agent.buffer_rl
            else:
                target_buffer = algo_agent.buffer
            target_buffer.push_batch(s, a, r, sn, d)


    def _print_td3_diagnostic(self):
        if not self.diag_enabled:
            return
        raw = np.asarray(self._diag_raw_actions, dtype=float)
        env_a = np.asarray(self._diag_env_actions, dtype=float)
        rew = np.asarray(self._diag_rewards, dtype=float)
        self._diag_episode += 1

        if env_a.size:
            v2g = float(np.mean(env_a <= -0.5))
            v2h = float(np.mean((env_a > -0.5) & (env_a <= 0.0)))
            idle = float(np.mean((env_a > 0.0) & (env_a <= 0.5)))
            ch = float(np.mean(env_a > 0.5))
        else:
            v2g = v2h = idle = ch = float("nan")

        q1 = self._diag_last_learn.get("q1_mean", float("nan"))
        q2 = self._diag_last_learn.get("q2_mean", float("nan"))
        tq = self._diag_last_learn.get("target_q_mean", float("nan"))
        cl = self._diag_last_learn.get("critic_loss", float("nan"))
        print(
            f"[{self.algo_name.upper()} DIAG] ep={self._diag_episode:03d} | "
            f"reward_mean={np.mean(rew) if rew.size else float('nan'):+.5f} | "
            f"raw={np.nanmean(raw) if raw.size else float('nan'):+.3f}±{np.nanstd(raw) if raw.size else float('nan'):.3f} "
            f"[{np.nanmin(raw) if raw.size else float('nan'):+.3f},{np.nanmax(raw) if raw.size else float('nan'):+.3f}] | "
            f"exec={np.mean(env_a) if env_a.size else float('nan'):+.3f}±{np.std(env_a) if env_a.size else float('nan'):.3f} "
            f"[{np.min(env_a) if env_a.size else float('nan'):+.3f},{np.max(env_a) if env_a.size else float('nan'):+.3f}] | "
            f"modes V2G={v2g:.1%} V2H={v2h:.1%} Idle={idle:.1%} CH={ch:.1%} | "
            f"Q1={q1:+.3f} Q2={q2:+.3f} targetQ={tq:+.3f} criticLoss={cl:.4g}"
        )
        self._diag_step_count = 0
        self._diag_pending_print = False
        self._diag_raw_actions.clear()
        self._diag_env_actions.clear()
        self._diag_rewards.clear()

    def learn(self, demo_frac=None):
        if demo_frac is None:
            demo_frac = self.demo_frac
        logs = []
        for _ in range(self.updates_per_step):
            m = self.role_algo["V2G"].learn_step(demo_frac)
            if m is not None:
                self.n_updates += 1
                logs.append(m)
                if self.diag_enabled:
                    self._diag_last_learn = dict(m)

        if self.diag_enabled and self._diag_pending_print:
            self._print_td3_diagnostic()
        return logs
