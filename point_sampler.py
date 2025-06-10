import torch
import numpy as np

from scipy.ndimage import distance_transform_edt
from monai.apps.vista3d.sampler import _get_point_label
import random

from scipy.ndimage import distance_transform_edt
from monai.apps.vista3d.sampler import _get_point_label

from scipy.ndimage import distance_transform_edt
from monai.apps.vista3d.sampler import _get_point_label

def gaussian_edge_center_sampler(
    unique_labels: list[int],
    *,
    gt_labels: torch.Tensor,      # [H,W,D]
    sampler_max_point: int,
    t_center: float = 0.5,
    t_edge:   float = 0.1,
    device:   torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    num_p = min(sampler_max_point, int(abs(random.gauss(0, sampler_max_point // 2))) + 1)
    num_n = min(sampler_max_point, int(abs(random.gauss(0, sampler_max_point // 2))))

    total = num_p + num_n
    points, point_labels = [], []

    for cid in unique_labels:
        mask_np = (gt_labels.cpu().numpy() == cid)
        dist = distance_transform_edt(mask_np)
        maxd = dist.max() if dist.max() > 0 else 1.0

        center_np = dist > (t_center * maxd)
        edge_np   = (dist > (t_edge * maxd)) & (dist <= (t_center * maxd))

        center_pts = torch.from_numpy(np.stack(np.nonzero(center_np), axis=1)).to(device)
        edge_pts   = torch.from_numpy(np.stack(np.nonzero(edge_np),   axis=1)).to(device)
        all_pos    = torch.nonzero(gt_labels == cid)

        if len(center_pts) > 0:
            c = center_pts[torch.randint(len(center_pts), (1,))]
        else:
            c = all_pos[torch.randint(len(all_pos), (1,))]
        k_edge = max(num_p - 1, 0)
        if k_edge > 0 and len(edge_pts) >= k_edge:
            e = edge_pts[torch.randint(len(edge_pts), (k_edge,))]
        elif k_edge > 0:
            e = all_pos[torch.randint(len(all_pos), (k_edge,))]
        else:
            e = torch.empty((0, 3), device=device, dtype=torch.long)
        pos = torch.cat([c, e], dim=0)
        if pos.shape[0] < num_p:
            pad_n = num_p - pos.shape[0]
            pos = torch.cat([pos, torch.zeros((pad_n, 3), device=device, dtype=torch.long)], dim=0)

        neg_region = torch.nonzero(gt_labels != cid)
        if len(neg_region) >= num_n:
            neg = neg_region[torch.randint(len(neg_region), (num_n,))]
        else:
            pad_n = num_n - len(neg_region)
            neg = torch.cat([neg_region, torch.zeros((pad_n, 3), device=device, dtype=torch.long)], dim=0)

        all_pts = torch.cat([pos, neg], dim=0)  # shape [num_p+num_n, 3]

        neg_id, pos_id = _get_point_label(cid)
        pl = (
            torch.full((num_p,), pos_id, device=device, dtype=torch.long)
            .tolist()
            + torch.full((num_n,), neg_id, device=device, dtype=torch.long).tolist()
        )
        pl = torch.tensor(pl + [-1] * (total - num_p - num_n), device=device, dtype=torch.long)

        points.append(all_pts)
        point_labels.append(pl)

    return points, point_labels

