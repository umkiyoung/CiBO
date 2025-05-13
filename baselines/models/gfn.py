import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gfn_folder.gfn_utils import gaussian_params
import torch
import numpy as np
from einops import rearrange
from torch import nn
import math
logtwopi = math.log(2 * math.pi)

class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers, dtype=torch.float32):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim),
        )
        self.dtype = dtype
        # Initialize the last layer's weights and biases
        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.to(dtype=self.dtype)) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.to(dtype=self.dtype)) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(
            self,
            in_shape=2,
            out_shape=2,
            num_layers=2,
            channels=128,
            zero_init=True,
            dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.to(dtype=self.dtype)) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.to(dtype=self.dtype)) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64, dtype=torch.float32):
        super(TimeEncoding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).to(dtype=dtype).unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = (t * self.pe).sin()
        t_cos = (t * self.pe).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncoding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU()
        )

    def forward(self, s):
        return self.x_model(s)


class JointPolicy(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 zero_init: bool = False):
        super(JointPolicy, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class FlowModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1):
        super(FlowModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class LangevinScalingModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, zero_init: bool = False):
        super(LangevinScalingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.01)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncodingPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncodingPIS, self).__init__()

        self.x_model = nn.Linear(s_dim, s_emb_dim)

    def forward(self, s):
        return self.x_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(JointPolicyPIS, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class FlowModelPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(FlowModelPIS, self).__init__()

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class LangevinScalingModelPIS(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinScalingModelPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)

class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., langevin: bool = False, learned_variance: bool = True,
                 trajectory_length: int = 100, partial_energy: bool = False,
                 clipping: bool = False, lgv_clip: float = 1e2, gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 langevin_scaling_per_dimension: bool = True, conditional_flow_model: bool = False,
                 learn_pb: bool = False,
                 pis_architectures: bool = False, lgv_layers: int = 3, joint_layers: int = 2,
                 zero_init: bool = False, device=torch.device('cuda'), dtype=torch.float32):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.trajectory_length = trajectory_length
        self.langevin = langevin
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.langevin_scaling_per_dimension = langevin_scaling_per_dimension
        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.pis_architectures = pis_architectures
        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device
        self.dtype = dtype
        if self.pis_architectures:

            self.t_model = TimeEncodingPIS(harmonics_dim, t_dim, hidden_dim).to(device=self.device, dtype=dtype)
            self.s_model = StateEncodingPIS(dim, hidden_dim, s_emb_dim).to(device=self.device, dtype=dtype)
            self.joint_model = JointPolicyPIS(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, joint_layers, zero_init).to(device=self.device, dtype=dtype)
            if learn_pb:
                self.back_model = JointPolicyPIS(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, joint_layers, zero_init).to(device=self.device, dtype=dtype)
            self.pb_scale_range = pb_scale_range

            if self.conditional_flow_model:
                self.flow_model = FlowModelPIS(dim, s_emb_dim, t_dim, hidden_dim, 1, joint_layers).to(device=self.device, dtype=dtype)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(device=self.device, dtype=dtype))

            if self.langevin_scaling_per_dimension:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, dim,
                                                                      lgv_layers, zero_init).to(device=self.device, dtype=dtype)
            else:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, 1,
                                                                      lgv_layers, zero_init).to(device=self.device, dtype=dtype)

        else:

            self.t_model = TimeEncoding(harmonics_dim, t_dim, hidden_dim, dtype=dtype).to(device=self.device, dtype=dtype)
            self.s_model = StateEncoding(dim, hidden_dim, s_emb_dim).to(device=self.device, dtype=dtype)
            self.joint_model = JointPolicy(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, zero_init).to(device=self.device, dtype=dtype)
            if learn_pb:
                self.back_model = JointPolicy(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, zero_init).to(device=self.device, dtype=dtype)
            self.pb_scale_range = pb_scale_range

            if self.conditional_flow_model:
                self.flow_model = FlowModel(s_emb_dim, t_dim, hidden_dim, 1).to(device=self.device, dtype=dtype)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(device=self.device, dtype=dtype))
            
            if self.langevin_scaling_per_dimension:
                self.langevin_scaling_model = LangevinScalingModel(s_emb_dim, t_dim, hidden_dim, dim, zero_init).to(device=self.device, dtype=dtype)
            else:
                self.langevin_scaling_model = LangevinScalingModel(s_emb_dim, t_dim, hidden_dim, 1, zero_init).to(device=self.device, dtype=dtype)

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def predict_next_state(self, s, t, log_r):
        if self.langevin:
            s.requires_grad_(True)
            with torch.enable_grad():
                grad_log_r = torch.autograd.grad(log_r(s, track_gradient=True).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        bsz = s.shape[0]
        t_lgv = t

        t = self.t_model(t).repeat(bsz, 1)
        s = self.s_model(s)
        s_new = self.joint_model(s, t)

        flow = self.flow_model(s, t).squeeze(-1) if self.conditional_flow_model or self.partial_energy else self.flow_model
        if self.langevin:
            if self.pis_architectures:
                scale = self.langevin_scaling_model(t_lgv)
            else:
                scale = self.langevin_scaling_model(s, t)
            s_new[..., :self.dim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    def get_trajectory_fwd(self, s, exploration_std, log_r, pis=False):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device, dtype=self.dtype)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device, dtype=self.dtype)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device, dtype=self.dtype)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device, dtype=self.dtype)

        for i in range(self.trajectory_length):
            pfs, flow = self.predict_next_state(s, i * self.dt, log_r)
            # pfs = self.predict_next_state(s, i * self.dt, log_r)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, i] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale * max(1, i) * self.dt)
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, i] += (1 - i * self.dt) * log_p_ref + i * self.dt * log_r(s)

            if exploration_std is None:
                if pis:
                    pflogvars_sample = pflogvars
                else:
                    pflogvars_sample = pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                    if pis:
                        pflogvars_sample = torch.logaddexp(pflogvars, add_log_var)
                    else:
                        pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()

            if pis:
                s_ = s + self.dt * pf_mean + np.sqrt(self.dt) * (
                        pflogvars_sample / 2).exp() * torch.randn_like(s, device=self.device)
            else:
                s_ = s + self.dt * pf_mean.detach() + np.sqrt(self.dt) * (
                        pflogvars_sample / 2).exp() * torch.randn_like(s, device=self.device)

            noise = ((s_ - s) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(1)

            if self.learn_pb:
                t = self.t_model((i + 1) * self.dt).repeat(bsz, 1)
                pbs = self.back_model(self.s_model(s_), t)
                dmean, dvar = gaussian_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction, back_var_correction = torch.ones_like(s_), torch.ones_like(s_)

            if i > 0:
                back_mean = s_ - self.dt * s_ / ((i + 1) * self.dt) * back_mean_correction
                back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction
                noise_backward = (s - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

            s = s_
            states[:, i + 1] = s

        return states, logpf, logpb, logf

    def get_trajectory_bwd(self, s, exploration_std, log_r):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device, dtype=self.dtype)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device, dtype=self.dtype)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device, dtype=self.dtype)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device, dtype=self.dtype)
        states[:, -1] = s

        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                if self.learn_pb:
                    t = self.t_model(1. - i * self.dt).repeat(bsz, 1)
                    pbs = self.back_model(self.s_model(s), t)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(s), torch.ones_like(s)

                mean = s - self.dt * s / (1. - i * self.dt) * back_mean_correction
                var = ((self.pf_std_per_traj ** 2) * self.dt * (1. - (i + 1) * self.dt)) / (
                            1 - i * self.dt) * back_var_correction
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)
            else:
                s_ = torch.zeros_like(s)

            pfs, flow = self.predict_next_state(s_, (1. - (i + 1) * self.dt), log_r)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, self.trajectory_length - i - 1] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale * max(1, self.trajectory_length - i - 1) * self.dt)
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, self.trajectory_length - i - 1] += (i + 1) * self.dt * log_p_ref + (
                        self.trajectory_length - i - 1) * self.dt * log_r(s)

            noise = ((s - s_) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(
                1)

            s = s_
            states[:, self.trajectory_length - i - 1] = s

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r):
        s = torch.zeros(batch_size, self.dim, device=self.device, dtype=self.dtype)
        return self.get_trajectory_fwd(s, None, log_r)[0][:, -1]
    
    # def sample_normal(self, initial_states, batch_size, log_r):
    #     # s = torch.zeros(batch_size, self.dim).to(self.device)
    #     return self.get_trajectory_fwd(initial_states, None, log_r)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std):
        s = torch.zeros(batch_size, self.dim, device=self.device, dtype=self.dtype)
        return self.get_trajectory_fwd(s, exploration_std, log_r=None)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r)
