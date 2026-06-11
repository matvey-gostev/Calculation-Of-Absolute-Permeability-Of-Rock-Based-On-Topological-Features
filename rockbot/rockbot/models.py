import pickle
import warnings

import numpy as np


class ModelLoadError(RuntimeError):
    pass


class PersistenceTransformerUnavailable(RuntimeError):
    pass


try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class PersistenceTransformer(nn.Module):
        def __init__(
            self,
            input_dim=3,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            max_len=2000,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            self.max_len = max_len

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, points, mask=None):
            batch_size, num_points, _ = points.shape
            if num_points > self.max_len:
                points = points[:, : self.max_len, :]
                if mask is not None:
                    mask = mask[:, : self.max_len]
                num_points = self.max_len

            x = self.input_proj(points) + self.pos_embed[:, :num_points, :]
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)
            if mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

            x = self.transformer(x, src_key_padding_mask=mask)
            return self.head(x[:, 0, :])

else:
    PersistenceTransformer = None


class TorchEnsemble:
    def __init__(self, spec, models, device, max_points):
        self.spec = spec
        self.models = models
        self.device = device
        self.max_points = max_points

    @classmethod
    def load(cls, spec, device_name=None):
        if torch is None or PersistenceTransformer is None:
            raise PersistenceTransformerUnavailable("Для модельного инференса нужен torch.")

        weight_paths = discover_weight_paths(spec.artifact_dir)
        if not weight_paths:
            raise ModelLoadError(f"В {spec.artifact_dir} не найдены .pth/.pt веса.")

        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        models = []
        model_max_points = []
        for path in weight_paths:
            state = _torch_load(path, device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            max_len = _infer_max_len(state, fallback=spec.max_points)
            model = PersistenceTransformer(
                d_model=128,
                nhead=4,
                num_layers=3,
                dim_feedforward=256,
                dropout=0.2,
                max_len=max_len,
            )
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
            model_max_points.append(max_len)
        max_points = min([*model_max_points, spec.max_points]) if model_max_points else spec.max_points
        return cls(spec=spec, models=models, device=device, max_points=max_points)

    def predict_log(self, points):
        if torch is None:
            raise PersistenceTransformerUnavailable("Для модельного инференса нужен torch.")
        points_t = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.zeros(1, points_t.shape[1], dtype=torch.bool, device=self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(points_t, mask_t).detach().cpu())
        return float(torch.stack(preds).mean().item())


def discover_weight_paths(model_dir):
    preferred = [model_dir / f"best_model_fold{i}.pth" for i in range(1, 6)]
    if all(path.exists() for path in preferred):
        return preferred
    paths = sorted(model_dir.glob("*.pth")) + sorted(model_dir.glob("*.pt"))
    return paths


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_max_len(state, fallback):
    if isinstance(state, dict):
        pos_embed = state.get("pos_embed")
        shape = getattr(pos_embed, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
    return fallback


def load_domain_classifier(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with path.open("rb") as file:
            classifier = pickle.load(file)
    _patch_old_sklearn_trees(classifier)
    return classifier


def _patch_old_sklearn_trees(classifier):
    estimators = getattr(classifier, "estimators_", None)
    if estimators is None:
        return
    for estimator in estimators:
        if not hasattr(estimator, "monotonic_cst"):
            setattr(estimator, "monotonic_cst", None)


def normalize_pd_points(points, norm_stats, max_points=2000):
    if points.size == 0:
        return None
    points = points.astype(np.float32, copy=True)
    for col_idx in (1, 2):
        col = points[:, col_idx]
        inf_mask = np.isinf(col)
        if inf_mask.any():
            finite = col[~inf_mask]
            max_finite = float(finite.max()) if len(finite) else 1.0
            points[inf_mask, col_idx] = max_finite * 1.1

    if points.shape[0] > max_points:
        persistence = points[:, 2] - points[:, 1]
        idx = np.argsort(np.abs(persistence))[::-1][:max_points]
        points = points[idx]

    points[:, 1] = (points[:, 1] - norm_stats["mean_birth"]) / (norm_stats["std_birth"] + 1e-8)
    points[:, 2] = (points[:, 2] - norm_stats["mean_death"]) / (norm_stats["std_death"] + 1e-8)
    if np.isnan(points).any() or np.isinf(points).any():
        return None
    return points
