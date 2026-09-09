# CVCDepth reproduction portability patch.
#
# The original CVCDepth package imports DGPDataset and
# SynchronizedSceneDataset here even though the custom DDAD loader
# only requires get_transforms() and stack_sample().
#
# This avoids importing the unused DGP runtime while preserving
# the original stack_sample behavior.

import numpy as np
import torch

from external.packnet_sfm.packnet_sfm.datasets.transforms import get_transforms


def stack_sample(sample):
    """Stack a sample from multiple sensors.

    Equivalent to the helper used in the original vendored
    PackNet-SfM dgp_dataset.py.
    """

    if len(sample) == 1:
        return sample[0]

    stacked_sample = {}

    for key in sample[0]:

        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]

        elif torch.is_tensor(sample[0][key]):
            stacked_sample[key] = torch.stack(
                [s[key] for s in sample], 0
            )

        elif isinstance(sample[0][key], np.ndarray):
            stacked_sample[key] = np.stack(
                [s[key] for s in sample], 0
            )

        elif isinstance(sample[0][key], list):
            stacked_sample[key] = []

            if len(sample[0][key]) == 0:
                continue

            if torch.is_tensor(sample[0][key][0]):
                for i in range(len(sample[0][key])):
                    stacked_sample[key].append(
                        torch.stack(
                            [s[key][i] for s in sample], 0
                        )
                    )

            elif isinstance(sample[0][key][0], np.ndarray):
                for i in range(len(sample[0][key])):
                    stacked_sample[key].append(
                        np.stack(
                            [s[key][i] for s in sample], 0
                        )
                    )

    return stacked_sample


__all__ = ['get_transforms', 'stack_sample']
