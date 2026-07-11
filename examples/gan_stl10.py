"""Multi-round LLM-babysat DCGAN on STL-10 at 64x64.

Vanilla DCGAN on STL-10 unlabeled images (resized to 64x64) with FID as the goal
metric. Higher resolution and diverse natural scenes make vanilla GAN training
genuinely unstable — mode collapse and D domination appear without artificial
misconfiguration. An LLM babysitter tunes lr_g / lr_d / n_critic and can
early-stop to capture the best FID.

Run (inside apptainer, after `source init.sh`):
    python -m examples.gan_stl10 --max-rounds 4            # babysat
    python -m examples.gan_stl10 --no-agent --max-rounds 1 # pure baseline
"""
from __future__ import annotations

import argparse
import logging
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import make_grid

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from interactive_training.recipes.gan import gan

setup_logging()
logger = logging.getLogger(__name__)

STL10_TRAIN_SIZE = 100_000


class Generator(nn.Module):
    """DCGAN generator for 64x64 RGB (one extra upsample vs the 32x32 variant)."""

    def __init__(self, nz: int = 100, ngf: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN discriminator for 64x64 RGB (one extra downsample vs the 32x32 variant)."""

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FIDScorer:
    """Inception-v3 pool features + Fréchet distance (no torchmetrics dependency)."""

    def __init__(self, device: str, real_loader: DataLoader, n_images: int):
        self.device = device
        self.n_images = n_images
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.aux_logits = False
        self.inception.eval().to(device)
        for p in self.inception.parameters():
            p.requires_grad_(False)
        self.mu_real, self.sigma_real = self._fit_real(real_loader)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _features(self, images: torch.Tensor) -> np.ndarray:
        feats = self.inception(self._preprocess(images))
        return feats.cpu().numpy()

    @staticmethod
    def _stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    @staticmethod
    def _frechet(mu1, sigma1, mu2, sigma2) -> float:
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

    def _fit_real(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        n = 0
        for batch, _ in loader:
            feats.append(self._features(batch.to(self.device)))
            n += batch.size(0)
            if n >= self.n_images:
                break
        arr = np.concatenate(feats, axis=0)[: self.n_images]
        logger.info("FID reference stats from %d real images", arr.shape[0])
        return self._stats(arr)

    @torch.no_grad()
    def score(self, G: nn.Module, nz: int, batch_size: int) -> float:
        G.eval()
        feats = []
        remaining = self.n_images
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, nz, 1, 1, device=self.device)
            fake = G(z)
            feats.append(self._features(fake))
            remaining -= bs
        G.train()
        mu, sigma = self._stats(np.concatenate(feats, axis=0))
        return self._frechet(self.mu_real, self.sigma_real, mu, sigma)


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _dataset_root() -> str:
    return os.path.join(os.environ.get("DATASET_BASE_PATH", "."), "data")


def make_dataloader(args) -> DataLoader:
    ds = datasets.STL10(
        root=_dataset_root(),
        split="unlabeled",
        download=True,
        transform=_image_transform(args.image_size),
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_fid_subset_loader(args, n: int, seed: int = DEFAULT_SEED) -> DataLoader:
    ds = datasets.STL10(
        root=_dataset_root(),
        split="test",
        download=True,
        transform=_image_transform(args.image_size),
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return DataLoader(
        Subset(ds, idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _steps_per_epoch(batch_size: int, n_train: int = STL10_TRAIN_SIZE) -> int:
    return n_train // batch_size


def build_context(args) -> str:
    spe = _steps_per_epoch(args.batch_size)
    epochs = args.max_steps / spe
    return (
        f"Task: unconditional image generation on STL-10 ({args.image_size}x{args.image_size} RGB, "
        f"resized from native 96x96). Training uses the 100k-image unlabeled split; FID compares "
        f"against {args.fid_n} held-out test images (lower is better), measured every "
        f"{args.eval_steps} iterations.\n"
        f"Model: deeper DCGAN for 64x64 — Generator (5-layer ConvTranspose stack, nz={args.nz}) "
        f"and Discriminator (5-layer Conv stack), both initialized with N(0, 0.02).\n"
        f"Algorithm: vanilla GAN with BCEWithLogitsLoss. Each iteration: "
        f"n_critic discriminator updates then one generator update. "
        f"{args.max_steps} iterations per round (~{epochs:.1f} epochs at batch size "
        f"{args.batch_size}). Diverse 64x64 natural scenes make vanilla GAN balancing harder "
        f"than on small fixed datasets — instability is expected, not misconfiguration.\n"
        f"Typical failures: mode collapse (sample grid looks repetitive), D domination "
        f"(D_x→1, D_G_z→0, g_loss climbs, fid stalls or rises), or oscillating quality. "
        f"FID may improve early then degrade when D overfits.\n"
        f"Health signals: D_x = mean sigmoid(D(real)) should stay near ~0.5–0.8; "
        f"D_G_z = mean sigmoid(D(fake)) should not collapse to 0 (D winning) or 1 (G winning "
        f"without quality). Rising d_loss with worsening fid usually means D is too strong.\n"
        f"Knob guidance: 'lr_g' / 'lr_d' are Adam learning rates (default 2e-4). If D_x→1 and "
        f"D_G_z→0, lower lr_d or raise lr_g. If training is unstable (loss spikes), lower both. "
        f"'n_critic' (default {args.n_critic}) is D steps per G step — raise when G wins too "
        f"easily (D_G_z high), lower when D dominates. Strongly prefer early stop via 'stop' "
        f"once fid stops improving or starts rising."
    )


def main(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("gan_stl10", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "gan_stl10")
    memory_path = args.memory_path or os.path.join(run_dir, "gan_stl10_memory.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = make_dataloader(args)
    fid_loader = make_fid_subset_loader(args, args.fid_n, seed=args.seed)
    fid_scorer = FIDScorer(device, fid_loader, args.fid_n)

    agent = None if args.no_agent else LLMAgent(every=args.agent_every, name=args.agent_model,
                                                client=build_client(args))
    session = TrainingSession(goal="fid", memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds,
                              context=build_context(args), seed=args.seed)
    use_wandb = not args.no_wandb
    wandb_group = f"{args.wandb_experiment}-{run_id}"
    criterion = nn.BCEWithLogitsLoss()
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device, generator=g)

    def train_round(session, ctx):
        G = Generator(args.nz).to(device)
        D = Discriminator().to(device)
        G.apply(weights_init)
        D.apply(weights_init)
        g_opt = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        cfg = {"n_critic": args.n_critic}
        gan(session, g_opt, d_opt, cfg)
        session.bind_model(G)
        session.plan_round(ctx)

        os.makedirs(f"{output_dir}/round_{ctx.round}", exist_ok=True)
        babysat = agent is not None
        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, group=wandb_group,
                             name=f"{wandb_group}-round{ctx.round}", reinit=True,
                             config={"round": ctx.round, "babysat": babysat,
                                     "dataset": "stl10", "image_size": args.image_size,
                                     "lr_g": args.lr_g, "lr_d": args.lr_d,
                                     "n_critic": args.n_critic, "batch_size": args.batch_size,
                                     "max_steps": args.max_steps, "nz": args.nz})

        data_iter = iter(train_loader)
        for it in range(args.max_steps):
            d_loss_sum = d_x_sum = d_gz_sum = 0.0
            n_d = int(cfg["n_critic"])

            for _ in range(n_d):
                try:
                    real, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    real, _ = next(data_iter)
                real = real.to(device)
                bs = real.size(0)
                real_labels = torch.ones(bs, device=device)
                fake_labels = torch.zeros(bs, device=device)

                D.zero_grad(set_to_none=True)
                out_real = D(real)
                loss_d_real = criterion(out_real, real_labels)
                loss_d_real.backward()
                d_x_sum += out_real.sigmoid().mean().item()

                noise = torch.randn(bs, args.nz, 1, 1, device=device)
                fake = G(noise)
                out_fake = D(fake.detach())
                loss_d_fake = criterion(out_fake, fake_labels)
                loss_d_fake.backward()
                d_gz_sum += out_fake.sigmoid().mean().item()
                d_opt.step()
                d_loss_sum += (loss_d_real + loss_d_fake).item()

            G.zero_grad(set_to_none=True)
            out_fake = D(fake)
            loss_g = criterion(out_fake, real_labels)
            loss_g.backward()
            g_opt.step()

            metrics = {
                "d_loss": d_loss_sum / n_d,
                "g_loss": float(loss_g.detach()),
                "D_x": d_x_sum / n_d,
                "D_G_z": d_gz_sum / n_d,
            }
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                metrics["fid"] = fid_scorer.score(G, args.nz, args.batch_size)
                if run is not None:
                    with torch.no_grad():
                        samples = G(fixed_noise).cpu()
                    grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
                    run.log({"samples": wandb.Image(grid)}, step=it)

            if run is not None:
                knob_log = {f"knob/{k}": float(cfg[k]) for k in cfg}
                knob_log["knob/lr_g"] = g_opt.param_groups[0]["lr"]
                knob_log["knob/lr_d"] = d_opt.param_groups[0]["lr"]
                run.log({**metrics, **knob_log}, step=it)

            logger.info(
                "round %d step %d | d_loss=%.3f g_loss=%.3f D_x=%.3f D_G_z=%.3f%s",
                ctx.round, it, metrics["d_loss"], metrics["g_loss"],
                metrics["D_x"], metrics["D_G_z"],
                f" fid={metrics['fid']:.2f}" if "fid" in metrics else "",
            )

            ctrl = session.step(metrics, step=it, act=babysat)
            if ctrl.stop:
                break

        torch.save({"G": G.state_dict(), "D": D.state_dict()},
                   f"{output_dir}/round_{ctx.round}/checkpoint.pt")
        if run is not None:
            run.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="min")
    base = next((r for r in session.memory.rounds if r["round"] == 0), None)
    print(f"Baseline (round 0) fid={base['score'] if base else float('nan'):.2f}")
    print(f"Best round {best['round']} fid={best['score']:.2f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-round LLM-babysat DCGAN on STL-10 (64x64)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed (re-applied each round for a reproducible baseline)")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64, help="output resolution (STL-10 resized)")
    p.add_argument("--max-steps", type=int, default=5000,
                   help="training iterations per round (~6.4 epochs at batch 128 on STL-10)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--nz", type=int, default=100, help="latent dimension")
    p.add_argument("--lr-g", type=float, default=2e-4, help="generator learning rate")
    p.add_argument("--lr-d", type=float, default=2e-4, help="discriminator learning rate")
    p.add_argument("--n-critic", type=int, default=1, help="discriminator steps per generator step")
    p.add_argument("--eval-steps", type=int, default=200,
                   help="compute FID every N iterations (sparser for long runs)")
    p.add_argument("--fid-n", type=int, default=2048, help="images used for FID statistics")
    p.add_argument("--agent-every", type=int, default=50,
                   help="agent acts every N iterations")
    p.add_argument("--no-agent", action="store_true", help="pure DCGAN baseline, no LLM babysitter")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/gan_stl10/<run-id>/gan_stl10")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/gan_stl10/<run-id>/gan_stl10_memory.jsonl")
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="gan_stl10")
    p.add_argument("--no-wandb", action="store_true")
    main(p.parse_args())
