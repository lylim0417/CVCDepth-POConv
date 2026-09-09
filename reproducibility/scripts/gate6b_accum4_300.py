import random
import time
import numpy as np
import torch

from utils import get_config
from models import VFDepthAlgo


CFG = "configs/repro/cvcdepth_ddad_single_gpu.yaml"

UPDATES = 300
ACCUM = 4
LOG_EVERY = 10

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

cfg = get_config(CFG, mode="train")

assert cfg["training"]["batch_size"] == 1

print("===== INITIALIZING =====")

model = VFDepthAlgo(cfg, 0)
model.set_train()

loader = model.train_dataloader()
iterator = iter(loader)

torch.cuda.reset_peak_memory_stats()

start = time.time()


def grad_norm(net):
    total = 0.0

    for p in net.parameters():
        if p.grad is None:
            continue

        g = p.grad.detach()

        if not torch.isfinite(g).all():
            raise RuntimeError(
                "NON-FINITE GRADIENT"
            )

        total += float(
            torch.sum(g.double() ** 2)
        )

    return total ** 0.5


def check_depth(outputs):
    result = []

    for cam in range(model.num_cams):

        d = outputs[
            ("cam", cam)
        ][
            ("depth", 0)
        ].detach()

        s = outputs[
            ("cam", cam)
        ][
            ("disp", 0)
        ].detach()

        result.append({
            "depth_med": float(d.median()),
            "depth_min": float(d.min()),
            "depth_max": float(d.max()),
            "disp_mean": float(s.mean()),
            "disp_min": float(s.min()),
            "disp_max": float(s.max()),
        })

    return result


print()
print(
    "effective global batch:",
    ACCUM
)

print(
    "optimizer updates:",
    UPDATES
)

print(
    "samples processed:",
    UPDATES * ACCUM
)

print()
print("===== RUN =====")


for update in range(1, UPDATES + 1):

    model.optimizer.zero_grad(
        set_to_none=True
    )

    loss_sums = {}
    last_outputs = None

    for micro in range(ACCUM):

        try:
            inputs = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs = next(iterator)

        outputs, losses = model.process_batch(
            inputs,
            0
        )

        total = losses["total_loss"]

        if not torch.isfinite(total):
            raise RuntimeError(
                f"NON-FINITE LOSS "
                f"update={update} micro={micro}"
            )

        # DDP-style gradient averaging.
        (total / ACCUM).backward()

        for key, value in losses.items():

            if not torch.is_tensor(value):
                continue

            if value.numel() != 1:
                continue

            loss_sums[key] = (
                loss_sums.get(key, 0.0)
                + float(value.detach()) / ACCUM
            )

        last_outputs = outputs

    depth_gn = grad_norm(
        model.models["depth_net"]
    )

    pose_gn = grad_norm(
        model.models["pose_net"]
    )

    if not np.isfinite(depth_gn):
        raise RuntimeError(
            "BAD DEPTH GRADIENT"
        )

    if not np.isfinite(pose_gn):
        raise RuntimeError(
            "BAD POSE GRADIENT"
        )

    model.optimizer.step()

    if (
        update == 1
        or update % LOG_EVERY == 0
        or update == UPDATES
    ):

        elapsed = time.time() - start

        print()
        print("=" * 72)

        print(
            f"UPDATE {update:03d}/{UPDATES}",
            f"samples={update * ACCUM}",
            f"elapsed={elapsed:.1f}s"
        )

        print("=" * 72)

        print(
            "total loss:",
            f"{loss_sums['total_loss']:.6f}"
        )

        for key in [
            "reproj_loss",
            "spatio_loss",
            "spatio_tempo_loss",
            "smooth",
            "spatial_depth_consistency_loss",
            "sp_tp_recon_con_loss",
        ]:
            if key in loss_sums:
                print(
                    f"{key:36s}",
                    f"{loss_sums[key]:.6f}"
                )

        print(
            "grad:",
            f"depth={depth_gn:.6e}",
            f"pose={pose_gn:.6e}"
        )

        stats = check_depth(
            last_outputs
        )

        print()
        print("depth/disparity:")

        for cam, s in enumerate(stats):

            print(
                f"cam {cam}:",
                f"dmed={s['depth_med']:.3f}m",
                f"dmin={s['depth_min']:.3f}",
                f"dmax={s['depth_max']:.3f}",
                f"disp_mean={s['disp_mean']:.4f}",
                f"disp_max={s['disp_max']:.4f}",
            )

        print()
        print(
            "CUDA peak MB:",
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        # Hard diagnostic failure, not training clipping.
        means = [
            s["disp_mean"]
            for s in stats
        ]

        # Lower-depth boundary collapse.
        if min(means) > 0.98:
            raise RuntimeError(
                "DEPTH COLLAPSE: "
                "all camera disparity means > 0.98"
            )

        # Opposite / very-far-depth boundary collapse.
        if max(means) < 0.02:
            raise RuntimeError(
                "DEPTH COLLAPSE: "
                "all camera disparity means < 0.02"
            )


print()
print("canonical temporal pose:")

for frame_id in [-1, 1]:

    T = last_outputs[
        ("cam", 0)
    ][
        ("cam_T_cam", 0, frame_id)
    ].detach()

    tnorm = torch.linalg.norm(
        T[:, :3, 3],
        dim=-1
    ).mean()

    print(
        f"frame {frame_id:+d}:",
        f"|t|={float(tnorm):.6f} m"
    )
print("=" * 72)
print("GATE 6A COMPLETE")
print("=" * 72)
