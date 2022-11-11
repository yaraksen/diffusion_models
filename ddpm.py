import torch
import torch.nn as nn


def forward_process_schedules(beta1, beta2, T):
    beta_t = beta1 + torch.arange(0, T + 1, dtype=torch.float32) / T * (beta2 - beta1)
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    sqrt_mab = torch.sqrt(1 - alpha_bar_t)
    return {
        "alpha_t": alpha_t,
        "inv_alpha": 1 / torch.sqrt(alpha_t),
        "sqrt_beta_t": torch.sqrt(beta_t),
        "alphabar_t": alpha_bar_t,
        "sqrtab": torch.sqrt(alpha_bar_t),
        "sqrtmab": sqrt_mab,
        "mab_over_sqrtmab": (1 - alpha_t) / sqrt_mab
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in forward_process_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = (self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise)
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))
        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0

        print()
        for i in range(self.n_T, 0, -1):
            print(f'reverse process timestep={i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)[:n_sample]
            x_i = x_i[:n_sample]
            x_i = (self.inv_alpha[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)
        return x_i
