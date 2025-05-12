import torch
from torch import nn, Tensor
from torchdiffeq import odeint
import math
from typing import Tuple

class ODEFunc(nn.Module):
    def __init__(self, net):
        super(ODEFunc, self).__init__()
        self.net = net

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        t_expanded = t.expand(x_t.shape[0], 1)
        return self.net(torch.cat((t_expanded, x_t), dim=-1))

    def divergence(self, t: Tensor, x_t: Tensor, noise: Tensor) -> Tensor:
        # Hutchinson's estimator: Tr(df/dx) ≈ ε^T df/dx ε
        x_t.requires_grad_(True)
        f = self.forward(t, x_t)
        vjp = torch.autograd.grad(
            outputs=f,
            inputs=x_t,
            grad_outputs=noise,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return (vjp * noise).sum(dim=1)  # shape: (batch,)


class ODEFuncWithLogDet(nn.Module):
    def __init__(self, odefunc: ODEFunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, t: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, logp = state
        noise = torch.randn_like(x)
        dx = self.odefunc(t, x)
        div = self.odefunc.divergence(t, x, noise)
        return dx, -div


class FlowModel(nn.Module):
    def __init__(self, x_dim: int = 2, hidden_dim: int = 512, step_size: int = 30, device: str = 'cpu', dtype: torch.dtype = torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, x_dim)
        ).to(device=device, dtype=dtype)
        self.x_dim = x_dim
        self.loss_function = nn.MSELoss()
        self.device = device
        self.dtype = dtype
        self.step_size = step_size
        self.odefunc = ODEFunc(self.net)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def sample(self, batch_size: int, track_gradient: bool = False) -> Tensor:
        x = torch.randn(batch_size, self.x_dim, device=self.device, dtype=self.dtype)
        time_steps = torch.linspace(0, 1.0, self.step_size + 1, device=self.device, dtype=self.dtype)
        if not track_gradient:
            with torch.no_grad():
                x_t = odeint(self.odefunc, x, time_steps, method='rk4')
        else:
            x_t = odeint(self.odefunc, x, time_steps, method='rk4')

        return x_t[-1]

    def inverse_sample(self, x: Tensor, track_gradient: bool = False) -> Tensor:
        time_steps = torch.linspace(1.0, 0.0, self.step_size + 1, device=self.device, dtype=self.dtype)

        if not track_gradient:
            with torch.no_grad():
                x_t = odeint(self.odefunc, x, time_steps, method='rk4')
        else:
            x_t = odeint(self.odefunc, x, time_steps, method='rk4')

        return x_t[-1]

    def sample_with_noise(self, x, track_gradient: bool = False):
        time_steps = torch.linspace(0, 1.0, self.step_size + 1, device=self.device, dtype=self.dtype)

        if not track_gradient:
            with torch.no_grad():                
                x_t = odeint(self.odefunc, x, time_steps, method='rk4')
        else:
            x_t = odeint(self.odefunc, x, time_steps, method='rk4')

        return x_t[-1]

    def compute_loss(self, x_1: Tensor) -> Tensor:
        x_0 = torch.randn_like(x_1, device=self.device, dtype=self.dtype)
        t = torch.rand(len(x_1), 1, device=self.device, dtype=self.dtype)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        loss = self.loss_function(self(x_t, t), dx_t)
        return loss

    def log_likelihood(self, x: Tensor) -> Tensor:
        time = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
        logp = torch.zeros(x.size(0), device=self.device, dtype=self.dtype)

        odefunc_with_logdet = ODEFuncWithLogDet(self.odefunc)

        z_t, logp_t = odeint(
            odefunc_with_logdet,
            (x, logp),
            time,
            method='rk4'
        )

        z0, logp = z_t[-1], logp_t[-1]

        logpz = -0.5 * (z0 ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return logpz + logp
