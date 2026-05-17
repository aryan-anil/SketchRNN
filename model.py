import math
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BivariateGaussianMixture:
    def __init__(
        self,
        pi_logits: torch.Tensor,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        rho_xy: torch.Tensor,
    ):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_distribution(self) -> int:
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        self.pi_logits /= temperature
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)

    def set_tempertaure(self, temperature: float):
        self.set_temperature(temperature)

    def get_distribution(self):
        sigma_x = torch.clamp(self.sigma_x, min=1e-5)
        sigma_y = torch.clamp(self.sigma_y, min=1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)

        mean = torch.stack([self.mu_x, self.mu_y], dim=-1)
        cov = torch.stack(
            [
                sigma_x**2,
                rho_xy * sigma_x * sigma_y,
                rho_xy * sigma_x * sigma_y,
                sigma_y**2,
            ],
            dim=-1,
        )
        cov = cov.view(*sigma_y.shape, 2, 2)

        # [batch, seq_len, n_mixtures]
        # multi_dist.sample().shape -> [batch, seq_len, n_mixtures, 2]
        multi_dist = torch.distributions.MultivariateNormal(loc=mean, 
                                                            covariance_matrix=cov)
        
        # [batch, seq_len, n_mixtures] -> [batch, seq_len, n_mixtures]
        # cat_dist.sample().shape -> [batch, seq_len]
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)
        return cat_dist, multi_dist


class EncoderLSTM(nn.Module):
    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        self.mu_head = nn.Linear(enc_hidden_size * 2, d_z)
        self.sigma_head = nn.Linear(enc_hidden_size * 2, d_z)

    def forward(self, inputs: torch.Tensor, state=None):
        _, (hidden, _) = self.lstm(inputs.float(), state)
        hidden = einops.rearrange(hidden, "fb b h -> b (fb h)")
        mu = self.mu_head(hidden)
        log_var = self.sigma_head(hidden)
        sigma = torch.exp(log_var / 2.0)
        z = mu + sigma * torch.randn_like(sigma)
        return z, mu, log_var


class DecoderLSTM(nn.Module):
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)
        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_distributions)
        self.q_head = nn.Linear(dec_hidden_size, 3)
        self.q_log_softmax = nn.LogSoftmax(-1)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if state is None:
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())

        z = z.unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs, state = self.lstm(torch.cat([x, z], dim=-1), state)
        q_logits = self.q_log_softmax(self.q_head(outputs))
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(
            self.mixtures(outputs),
            self.n_distributions,
            2,
        )

        dist = BivariateGaussianMixture(
            pi_logits,
            mu_x,
            mu_y,
            torch.exp(sigma_x),
            torch.exp(sigma_y),
            torch.tanh(rho_xy),
        )
        return dist, q_logits, state
                     

def sample_step(dist, q_log_probs, temperature):
    dist.set_temperature(temperature)
    cat_dist, multi_dist = dist.get_distribution()
    mixture_idx = cat_dist.sample()
    xy_samples = multi_dist.sample()
    xy = xy_samples.gather(-2, mixture_idx[..., None, None].expand(*mixture_idx.shape, 1, 2)).squeeze(-2)

    pen_logits = q_log_probs / temperature
    pen = torch.distributions.Categorical(probs=F.softmax(pen_logits, dim=-1)).sample()
    return xy.squeeze(0).squeeze(0), int(pen.item())


def sample_sketch(decoder, d_z, scale, max_len, temperature, device):
    z = torch.randn(1, d_z, device=device)
    prev = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32, device=device).view(1, 1, 5)
    state = None
    strokes = []

    with torch.no_grad():
        for _ in range(max_len):
            dist, q_log_probs, state = decoder(prev, z, state)
            xy, pen = sample_step(dist, q_log_probs, temperature)

            if pen == 2:
                break

            dx = float(xy[0].item() * scale)
            dy = float(xy[1].item() * scale)
            pen_up = 1.0 if pen == 1 else 0.0
            strokes.append([dx, dy, pen_up])
            prev = torch.tensor([xy[0].item(), xy[1].item(), 1.0 - pen_up, pen_up, 0.0], device=device).view(1, 1, 5)

    return np.asarray(strokes, dtype=np.float32)