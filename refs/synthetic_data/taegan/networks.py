from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import init


def _weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Norm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

    elif classname.find("Linear") != -1:
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)


class CombU(nn.Module):
    def __init__(self, size: int, negative_slope: float = 0.2):
        super().__init__()
        relu_size = int(size * 0.5)
        elu_size = int(size * 0.25)
        log_size = size - relu_size - elu_size

        candidates = np.arange(size)
        taken = np.zeros_like(candidates, dtype=np.bool_)
        relu_dims = np.random.choice(candidates[~taken], relu_size, replace=False)
        taken[relu_dims] = True
        elu_dims = np.random.choice(candidates[~taken], elu_size, replace=False)
        taken[elu_dims] = True
        log_dims = np.random.choice(candidates[~taken], log_size, replace=False)
        taken[log_dims] = True
        relu_dims = torch.from_numpy(relu_dims)
        elu_dims = torch.from_numpy(elu_dims)
        log_dims = torch.from_numpy(log_dims)

        self.relu = nn.ReLU(inplace=False)
        self.elu = nn.ELU(inplace=False)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=False)

        self.register_buffer("relu_dims", relu_dims)
        self.register_buffer("elu_dims", elu_dims)
        self.register_buffer("log_dims", log_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out_x = [
            self.lrelu(x[:, self.relu_dims]),
            self.elu(x[:, self.elu_dims]),
            torch.log(self.relu(x[:, self.log_dims]) + 1)
        ]
        x = torch.cat(out_x, dim=1)
        x = x.view(*shape).contiguous()
        return x


class EncoderOutput:
    def __init__(self,
                 out: torch.Tensor,
                 info: Optional[torch.Tensor] = None):
        self.out = out
        self.info = info


class Encoder(nn.Module):
    def __init__(self,
                 in_dim: int, out_dim: int,
                 hidden_dim: int = 256, n_layers: int = 6,
                 act: bool = True):
        super().__init__()
        layers = []
        in_dims = [in_dim] + [hidden_dim] * (n_layers - 1)
        out_dims = [hidden_dim] * (n_layers - 1) + [out_dim]
        for i, o in zip(in_dims[:-1], out_dims[:-1]):
            layers.append(nn.Sequential(
                nn.Linear(i, o), CombU(o, negative_slope=0.2),
                nn.LayerNorm(o), nn.Dropout(0.2)
            ))
        if act:
            act_layer = nn.Identity()
        else:
            act_layer = nn.ReLU(True)
        layers.append(nn.Sequential(nn.Linear(in_dims[-1], out_dims[-1]), act_layer))
        self.layers = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor, with_info: bool = False) -> EncoderOutput:
        if with_info:
            out_info = None
            for i, module in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    out_info = x
                x = module(x)
            return EncoderOutput(x, out_info)
        else:
            return EncoderOutput(self.layers(x))


class Decoder(nn.Module):
    def __init__(self,
                 in_dim: int, out_dim: int,
                 hidden_dim: int = 256, n_layers: int = 6,):
        super().__init__()
        layers = []
        in_dims = [in_dim] + [hidden_dim] * (n_layers - 1)
        out_dims = [hidden_dim] * (n_layers - 1) + [out_dim]
        for i, o in zip(in_dims[:-1], out_dims[:-1]):
            layers.append(nn.Sequential(
                nn.Linear(i, o), CombU(o, negative_slope=0.0),
                nn.LayerNorm(o), nn.Dropout(0.2)
            ))
        layers.append(nn.Sequential(
            nn.Linear(in_dims[-1], out_dims[-1])
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


SPANS = List[Tuple[int, Literal["onehot", "normal"]]]


class AEOutput:
    def __init__(self,
                 out: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        self.out = out
        self.mask = mask


class AutoEncoder(nn.Module):
    def __init__(self,
                 data_dim: int, span_info: SPANS,
                 embed_dim: int = 256, cat_noise_dim: int = 64, cont_noise_dim: int = 64,
                 hidden_dim: int = 256, n_layers: int = 6):
        super().__init__()
        self.span_info = span_info
        self.encoder = Encoder(
            in_dim=data_dim + len(span_info), out_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers
        )
        self.decoder = Decoder(
            in_dim=embed_dim + cont_noise_dim + cat_noise_dim, out_dim=data_dim,
            hidden_dim=hidden_dim, n_layers=n_layers
        )
        self._cat_noise_dim = cat_noise_dim
        self._cont_noise_dim = cont_noise_dim

    def forward(self,
                known: torch.FloatTensor, known_indicator: torch.BoolTensor, include_mask: bool = False) -> AEOutput:
        mask = torch.zeros_like(known)
        st = 0
        for i, (w, _) in enumerate(self.span_info):
            mask[:, st:st + w] = known_indicator[:, i:i + 1]
            st += w
        mask = mask.bool()
        known = known.masked_fill(~mask.clone(), 0)

        data = torch.cat([known, known_indicator.float().clone()], dim=-1)
        encoded = self.encoder(data).out

        cont_noise = torch.randn(known.shape[0], self._cont_noise_dim, device=known.device, dtype=known.dtype)
        cat_noise = torch.randint(
            0, 2, size=(known.shape[0], self._cat_noise_dim), device=known.device, dtype=known.dtype
        )
        decoded = self.decoder(torch.cat([cat_noise, cont_noise, encoded], dim=-1))
        if include_mask:
            return AEOutput(decoded, mask)
        return AEOutput(decoded)


class ReconLossOutput:
    def __init__(self, losses: torch.FloatTensor, weights: torch.FloatTensor):
        self.losses = losses
        self.weights = weights

    @property
    def loss(self) -> torch.FloatTensor:
        return (self.losses * self.weights).sum() / self.losses.shape[0]


class TableReconLoss(nn.Module):
    def __init__(self, span_info: SPANS, min_weight: float = 0.1, max_weight: float = 1.0):
        super().__init__()
        self.span_info = span_info
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self._min_weight = min_weight
        self._max_weight = max_weight

    def forward(self,
                data: torch.FloatTensor, target: torch.FloatTensor, known_indicator: torch.BoolTensor
                ) -> ReconLossOutput:
        unknown_weights = ((known_indicator.sum(dim=-1) / known_indicator.shape[1]) *
                           (self._max_weight - self._min_weight) + self._min_weight)
        loss_weights = torch.ones_like(known_indicator, dtype=torch.float)
        loss_weights = (loss_weights * known_indicator.float() +
                        unknown_weights.unsqueeze(-1) * (~known_indicator).float())

        st = 0
        losses = torch.zeros_like(known_indicator, dtype=torch.float)
        for i, (w, t) in enumerate(self.span_info):
            if t == "onehot":
                col_loss = self.ce(data[:, st:st + w], target[:, st:st + w].argmax(dim=-1))
            elif t == "normal":
                col_loss = self.smooth_l1(torch.tanh(data[:, st:st + w]), target[:, st:st + w])[:, 0]
            else:
                raise ValueError(f"Span type {t} is invalid.")
            losses[:, i] = col_loss
            st += w

        return ReconLossOutput(losses=losses, weights=loss_weights)


class InteractionFactor(nn.Module):
    def __init__(self, span_info: SPANS):
        super().__init__()
        self.span_info = span_info
        st = 0
        self.all_st = []
        self.all_ed = []
        for w, _ in self.span_info:
            self.all_st.append(st)
            st += w
            self.all_ed.append(st)

    def forward(self, data: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.Tensor]:
        all_mean = []
        all_std = []
        for i in range(len(self.span_info) - 1):
            col = data[:, self.all_st[i]:self.all_ed[i]].unsqueeze(2)
            for j in range(i + 1, len(self.span_info)):
                ocol = data[:, self.all_st[j]:self.all_ed[j]]
                all_mean.append(
                    (col * ocol.unsqueeze(1)).view(data.shape[0], -1).mean(dim=0)
                )
                all_std.append(
                    (col * ocol.unsqueeze(1)).view(data.shape[0], -1).std(dim=0)
                )
        return torch.cat(all_mean, dim=-1), torch.cat(all_std, dim=-1)


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.FloatTensor, temperature: Union[float, torch.FloatTensor] = 1) -> torch.FloatTensor:
        return nn.functional.gumbel_softmax(
            x / temperature, tau=0.2, hard=True, dim=-1
        )


class TableActivation(nn.Module):
    def __init__(self, span_info: SPANS):
        super().__init__()
        self.span_info = span_info
        self.softmax = GumbelSoftmax()
        self.tanh = nn.Tanh()

    def forward(self, data: torch.FloatTensor, temperature: Union[float, torch.FloatTensor] = 1) -> torch.FloatTensor:
        st = 0
        new_data = torch.empty_like(data)
        for w, t in self.span_info:
            if t == "onehot":
                new_data[:, st:st + w] = self.softmax(data[:, st:st + w], temperature)
            else:
                new_data[:, st:st + w] = self.tanh(data[:, st:st + w])
            st += w
        return new_data


class GradientPenalty(nn.Module):
    def __init__(self,
                 discriminator: Encoder,
                 lambda_: float = 10.,
                 eps: float = 1e-10):
        super().__init__()
        self._discriminator = discriminator
        self.lambda_ = lambda_
        self._eps = eps

    def forward(self,
                real: torch.FloatTensor, fake: torch.FloatTensor) -> torch.FloatTensor:
        if self.lambda_ == 0:
            return real[0, 0].clone() * 0
        batch_size = real.shape[0]
        alpha = torch.rand(batch_size, 1, device=real.device)
        interpolates = self._slerp(alpha, real, fake)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self._discriminator(interpolates).out

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.lambda_
        return gradient_penalty

    def _slerp(self, val: torch.FloatTensor, low: torch.FloatTensor, high: torch.FloatTensor) -> torch.FloatTensor:
        low_norm = low / torch.norm(low, dim=1, keepdim=True).clamp(min=self._eps)
        high_norm = high / torch.norm(high, dim=1, keepdim=True).clamp(min=self._eps)
        norm = (low_norm * high_norm).sum(1).clamp(min=-1, max=1)
        omega = torch.acos(norm).view(val.size(0), 1)
        so = torch.sin(omega).clamp(min=self._eps)
        res = (torch.sin((1.0 - val) * omega) / so) * low + (
                torch.sin(val * omega) / so
        ) * high
        return res
