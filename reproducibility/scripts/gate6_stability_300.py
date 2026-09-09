import random
import time
import numpy as np
import torch

from utils import get_config
from models import VFDepthAlgo


CFG = "configs/repro/cvcdepth_ddad_single_gpu.yaml"
NUM_STEPS = 300
LOG_EVERY = 25
OBS_EVERY = 50

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

cfg = get_config(CFG, mode="train")

print("===== INITIALIZING CVCDEPTH =====")

model = VFDepthAlgo(cfg, 0)
model.set_train()

loader = model.train_dataloader()
iterator = iter(loader)

torch.cuda.reset_peak_memory_stats()

loss_history = []
start_time = time.time()


def grad_norm(net):
    params = [
        p for p in net.parameters()
        if p.requires_grad
    ]

    norm = torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=float("inf"),
        error_if_nonfinite=True,
    )

    return float(norm)


def parameters_finite(net):
    return all(
        torch.isfinite(p).all().item()
        for p in net.parameters()
    )


@torch.no_grad()
def spatial_support(inputs, outputs):
    vals = {}

    for target_cam in range(model.num_cams):

        target_depth = outputs[
            ("cam", target_cam)
        ][
            ("depth", 0)
        ]

        target_invK = inputs[
            ("inv_K", 0)
        ][:, target_cam]

        target_ext = inputs[
            "extrinsics"
        ][:, target_cam]

        target_mask = inputs[
            "mask"
        ][:, target_cam]

        for source_cam in model.rel_cam_list[target_cam]:

            source_K = inputs[
                ("K", 0)
            ][:, source_cam]

            source_ext_inv = inputs[
                "extrinsics_inv"
            ][:, source_cam]

            source_mask = inputs[
                "mask"
            ][:, source_cam]

            source_rgb = inputs[
                ("color", 0, 0)
            ][:, source_cam]

            T_source_from_target = (
                source_ext_inv @ target_ext
            )

            _, warped_mask = (
                model.view_rendering.get_virtual_image(
                    source_rgb,
                    source_mask,
                    target_depth,
                    target_invK,
                    source_K,
                    T_source_from_target,
                    0,
                )
            )

            valid = (
                (warped_mask > 0).float()
                * target_mask
            )

            vals[(target_cam, source_cam)] = (
                float(valid.mean())
            )

    arr = np.asarray(list(vals.values()))

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "0<-1": vals[(0, 1)],
        "0<-2": vals[(0, 2)],
        "2<-0": vals[(2, 0)],
    }


print()
print("===== START 300-STEP STABILITY RUN =====")

for step in range(1, NUM_STEPS + 1):

    try:
        inputs = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs = next(iterator)

    model.optimizer.zero_grad(set_to_none=True)

    outputs, losses = model.process_batch(
        inputs,
        0
    )

    total = losses["total_loss"]

    if not torch.isfinite(total):
        raise RuntimeError(
            f"NON-FINITE TOTAL LOSS AT STEP {step}: {total}"
        )

    total.backward()

    depth_gn = grad_norm(
        model.models["depth_net"]
    )

    pose_gn = grad_norm(
        model.models["pose_net"]
    )

    if not np.isfinite(depth_gn) or depth_gn <= 0:
        raise RuntimeError(
            f"BAD DEPTH GRADIENT AT STEP {step}: {depth_gn}"
        )

    if not np.isfinite(pose_gn) or pose_gn <= 0:
        raise RuntimeError(
            f"BAD POSE GRADIENT AT STEP {step}: {pose_gn}"
        )

    model.optimizer.step()

    loss_history.append(
        float(total.detach())
    )

    should_log = (
        step == 1
        or step % LOG_EVERY == 0
        or step == NUM_STEPS
    )

    if should_log:

        if not parameters_finite(
            model.models["depth_net"]
        ):
            raise RuntimeError(
                f"NON-FINITE DEPTH PARAMETERS AT STEP {step}"
            )

        if not parameters_finite(
            model.models["pose_net"]
        ):
            raise RuntimeError(
                f"NON-FINITE POSE PARAMETERS AT STEP {step}"
            )

        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print(
            f"STEP {step:04d}/{NUM_STEPS}",
            f"elapsed={elapsed:.1f}s",
            f"sec/step={elapsed/step:.3f}",
        )
        print("=" * 70)

        print(
            "loss:",
            f"{float(total.detach()):.6f}"
        )

        for name in [
            "reproj_loss",
            "spatio_loss",
            "spatio_tempo_loss",
            "smooth",
            "spatial_depth_consistency_loss",
            "sp_tp_recon_con_loss",
        ]:
            if name in losses:
                print(
                    f"{name:36s}",
                    f"{float(losses[name]):.6f}"
                )

        print(
            "grad norms:",
            f"depth={depth_gn:.6e}",
            f"pose={pose_gn:.6e}",
        )

        print()
        print("depth / disparity:")

        for cam in range(model.num_cams):

            depth = outputs[
                ("cam", cam)
            ][
                ("depth", 0)
            ].detach()

            disp = outputs[
                ("cam", cam)
            ][
                ("disp", 0)
            ].detach()

            print(
                f"  cam {cam}:",
                f"depth_med={float(depth.median()):.3f}m",
                f"depth_min={float(depth.min()):.3f}",
                f"depth_max={float(depth.max()):.3f}",
                f"disp_mean={float(disp.mean()):.4f}",
                f"disp_min={float(disp.min()):.4f}",
                f"disp_max={float(disp.max()):.4f}",
            )

        print()
        print("canonical/front temporal pose:")

        for frame_id in [-1, 1]:

            T = outputs[
                ("cam", 0)
            ][
                ("cam_T_cam", 0, frame_id)
            ].detach()

            tnorm = torch.linalg.norm(
                T[:, :3, 3],
                dim=-1
            ).mean()

            print(
                f"  frame {frame_id:+d}:",
                f"|t|={float(tnorm):.6f} m"
            )

        if (
            step == 1
            or step % OBS_EVERY == 0
            or step == NUM_STEPS
        ):
            obs = spatial_support(
                inputs,
                outputs
            )

            print()
            print("spatial observability:")

            print(
                "  all edges:",
                f"min={obs['min']:.4f}",
                f"median={obs['median']:.4f}",
                f"mean={obs['mean']:.4f}",
                f"max={obs['max']:.4f}",
            )

            print(
                "  selected:",
                f"0<-1={obs['0<-1']:.4f}",
                f"0<-2={obs['0<-2']:.4f}",
                f"2<-0={obs['2<-0']:.4f}",
            )

        print()
        print(
            "CUDA MB:",
            f"allocated={torch.cuda.memory_allocated()/1024**2:.1f}",
            f"reserved={torch.cuda.memory_reserved()/1024**2:.1f}",
            f"peak={torch.cuda.max_memory_allocated()/1024**2:.1f}",
        )

    del outputs
    del losses


print()
print("=" * 70)
print("===== 300-STEP SUMMARY =====")
print("=" * 70)

losses_np = np.asarray(loss_history)

first = losses_np[:50]
last = losses_np[-50:]

print(
    "first-50 total loss:",
    f"mean={first.mean():.6f}",
    f"std={first.std():.6f}",
)

print(
    "last-50 total loss:",
    f"mean={last.mean():.6f}",
    f"std={last.std():.6f}",
)

print(
    "overall loss:",
    f"min={losses_np.min():.6f}",
    f"max={losses_np.max():.6f}",
)

print(
    "peak CUDA MB:",
    f"{torch.cuda.max_memory_allocated()/1024**2:.1f}",
)

print()
print("GATE 6 STABILITY RUN PASS")
