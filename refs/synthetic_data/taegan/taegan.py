import json
import os
from typing import Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from dataset import InferenceDataLoader, TrainingDataLoader
from networks import (
    AutoEncoder, Encoder, InteractionFactor, GradientPenalty, TableActivation, TableReconLoss
)


class TAEGAN:
    def __init__(self,
                 cache_dir: str,
                 embed_dim: int = 256, cat_noise_dim: int = 64, cont_noise_dim: int = 64,
                 hidden_dim: int = 256, n_layers: int = 6
                 ):
        with open(os.path.join(cache_dir, "span-info.json"), "r") as f:
            self.span_info = json.load(f)
        data_dim = sum(w for w, _ in self.span_info)
        self.generator = AutoEncoder(
            data_dim=data_dim, span_info=self.span_info,
            embed_dim=embed_dim, cat_noise_dim=cat_noise_dim, cont_noise_dim=cont_noise_dim,
            hidden_dim=hidden_dim, n_layers=n_layers
        )
        self.discriminator = Encoder(
            in_dim=data_dim + len(self.span_info), out_dim=1, act=False,
            hidden_dim=hidden_dim, n_layers=n_layers
        )
        self.act = TableActivation(self.span_info)
        self.cache_dir = cache_dir

    def train(self,
              batch_size: int = 500, epochs: int = 300, warmup_epochs: int = 90,
              lr: float = 2e-4, l2scale: float = 1e-5,
              discrimination_steps: int = 2, generation_steps: int = 1, reconstruction_steps: int = 2,
              min_recon_weight: float = 0.1, max_recon_weight: float = 1.0, gp_lambda: float = 10):
        dataloader = TrainingDataLoader(self.cache_dir, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.act = self.act.to(device)

        g_opt = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9), eps=1e-3, weight_decay=l2scale)
        d_opt = Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9), eps=1e-3, weight_decay=l2scale)
        g_scheduler = ExponentialLR(g_opt, gamma=0.99)
        d_scheduler = ExponentialLR(d_opt, gamma=0.99)

        gp = GradientPenalty(self.discriminator, gp_lambda)
        int_factor = InteractionFactor(self.span_info)
        loss_fct = TableReconLoss(
            span_info=self.span_info, min_weight=min_recon_weight, max_weight=max_recon_weight
        )

        pbar = tqdm(desc="Warmup Train", total=warmup_epochs * len(dataloader))
        for i in range(warmup_epochs):
            for d, m in dataloader:
                d = d.to(device)
                m = m.to(device)

                for _ in range(reconstruction_steps):
                    self._run_reconstruction_step(d, m, g_opt, loss_fct)
                for _ in range(discrimination_steps):
                    self._run_discrimination_step(d, m, d_opt, gp)
                pbar.update()
            d_scheduler.step()
            g_scheduler.step()

        pbar = tqdm(desc="Train", total=epochs * len(dataloader))
        for i in range(epochs):
            for d, m in dataloader:
                d = d.to(device)
                m = m.to(device)

                for _ in range(discrimination_steps):
                    self._run_discrimination_step(d, m, d_opt, gp)
                for _ in range(generation_steps):
                    self._run_generation_step(d, m, g_opt, int_factor, loss_fct)
                for _ in range(reconstruction_steps):
                    self._run_reconstruction_step(d, m, g_opt, loss_fct)
                pbar.update()
            d_scheduler.step()
            g_scheduler.step()

        torch.save(self.generator.state_dict(), os.path.join(self.cache_dir, "generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.cache_dir, "discriminator.pt"))

    def _run_reconstruction_step(self,
                                 d: torch.FloatTensor, m: torch.BoolTensor, g_opt: Adam, loss_fct: TableReconLoss):
        g_opt.zero_grad()
        perf_f = torch.randperm(d.shape[0])
        perf_f_d = d[perf_f]
        perf_f_m = m[perf_f]
        fake = self.generator(perf_f_d, perf_f_m).out
        recon_loss = loss_fct(fake, perf_f_d, perf_f_m)
        recon_loss.loss.backward()
        g_opt.step()

    def _run_discrimination_step(self,
                                 d: torch.FloatTensor, m: torch.BoolTensor, d_opt: Adam, gp: GradientPenalty):
        d_opt.zero_grad()
        perf_f = torch.randperm(d.shape[0])
        pref_r = torch.randperm(d.shape[0])
        perf_f_d = d[perf_f]
        perf_f_m = m[perf_f]
        fake = self.generator(perf_f_d, perf_f_m).out
        fakeact = self.act(fake).clone()

        real_cat = torch.cat([d[pref_r], m[perf_f].float()], dim=1)
        d_real = self.discriminator(real_cat).out
        d_real = -torch.mean(d_real)
        d_real.backward()
        fake_cat = torch.cat([fakeact, perf_f_m.float()], dim=1)
        d_fake = self.discriminator(fake_cat).out
        d_fake = torch.mean(d_fake)
        d_fake.backward()
        pen = gp(real_cat, fake_cat)
        pen.backward()
        d_opt.step()

    def _run_generation_step(self,
                             d: torch.FloatTensor, m: torch.BoolTensor, g_opt: Adam, int_factor: InteractionFactor,
                             loss_fct: TableReconLoss):
        g_opt.zero_grad()
        perf_f = torch.randperm(d.shape[0])
        pref_r = torch.randperm(d.shape[0])
        perf_f_d = d[perf_f]
        perf_f_m = m[perf_f]
        perf_r_d = d[pref_r]
        fake = self.generator(perf_f_d, perf_f_m).out
        fakeact = self.act(fake)

        real_cat = torch.cat([perf_r_d, m[pref_r].float()], dim=1)
        fake_cat = torch.cat([fakeact, perf_f_m.float()], dim=1)
        g_out = self.discriminator(fake_cat, with_info=True)
        g = g_out.out
        info_fake = g_out.info
        info_real = self.discriminator(real_cat, with_info=True).info
        g = -torch.mean(g)
        recon_loss = loss_fct(fake, perf_f_d, perf_f_m)
        loss_info_mean = torch.norm(torch.mean(info_fake, dim=0) - torch.mean(info_real, dim=0), 1)
        loss_info_std = torch.norm(torch.mean(info_fake, dim=0) - torch.std(info_real, dim=0), 1)
        real_int_mean, real_int_std = int_factor(perf_r_d)
        fake_int_mean, fake_int_std = int_factor(fakeact)
        loss_int_mean = torch.norm(real_int_mean - fake_int_mean, 1) / real_int_mean.shape[0]
        loss_int_std = torch.norm(real_int_std - fake_int_std, 1) / real_int_std.shape[0]

        gen_loss = loss_info_mean + loss_info_std + g + recon_loss.loss + loss_int_mean + loss_int_std
        gen_loss.backward()
        g_opt.step()

    @torch.no_grad()
    def generate(self,
                 n: int, batch_size: int = 100, temperature: Tuple[float, float] = (0.5, 1.2)) -> torch.FloatTensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = InferenceDataLoader(n, self.cache_dir, batch_size=batch_size)
        self.generator.load_state_dict(torch.load(os.path.join(self.cache_dir, "generator.pt")))
        self.generator = self.generator.to(device)
        self.generator.eval()
        self.act.eval()
        min_temp, max_temp = temperature

        pbar = tqdm(desc="Sampling", total=n)
        all_data = []
        for d, m in dataloader:
            d = d.to(device)
            m = m.to(device)
            for sm in m.permute(1, 0, 2):
                fake = self.generator(d, sm, True)
                mask = fake.mask
                fake = fake.out
                sm_ratio = sm.float().mean(dim=1).view(-1, 1)
                this_temp = (max_temp - min_temp) * (1 - sm_ratio) + min_temp
                fakeact = self.act(fake, temperature=this_temp)
                mask = mask.float()
                d = fakeact * (1 - mask) + d * mask
            all_data.append(d)
            pbar.update(d.shape[0])

        return torch.cat(all_data, dim=0).cpu()
