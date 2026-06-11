import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier_real_vs_synth.pkl"
RUNS_DIR = ARTIFACTS_DIR / "runs"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
PROJECT_VISUALS_DIR = ARTIFACTS_DIR / "project_visuals"

DEFAULT_TARGET_SHAPE = (256, 256, 256)
DEFAULT_FAST_FILTERS = (
    "radial_1_1_1",
    "height_z1",
    "height_z2",
    "line_Z1",
    "line_diag1_1_1_1",
)

REAL_NORM_STATS = {
    "mean_birth": 105.65832279905014,
    "std_birth": 71.82864306269713,
    "mean_death": 182.39315340882985,
    "std_death": 5561.2366693900785,
}

SYNTHETIC_NORM_STATS = {
    "mean_birth": 105.25917776343834,
    "std_birth": 64.71468071839826,
    "mean_death": 173.9171117042842,
    "std_death": 93.51994542248886,
}

MIXED_NORM_STATS = {
    key: (REAL_NORM_STATS[key] + SYNTHETIC_NORM_STATS[key]) / 2.0
    for key in REAL_NORM_STATS
}


class ModelSpec:
    def __init__(self, key, title, description, artifact_dir, norm_stats, demo_scale, metrics, max_points=2000):
        self.key = key
        self.title = title
        self.description = description
        self.artifact_dir = artifact_dir
        self.norm_stats = norm_stats
        self.demo_scale = demo_scale
        self.metrics = metrics
        self.max_points = max_points

    @property
    def expected_weight_paths(self):
        return [self.artifact_dir / f"best_model_fold{i}.pth" for i in range(1, 6)]

    @property
    def available_weight_paths(self):
        return sorted(self.artifact_dir.glob("*.pth")) + sorted(self.artifact_dir.glob("*.pt"))


class Settings:
    def __init__(self, telegram_token, target_shape, max_upload_mb, strict_model, fast_filters):
        self.telegram_token = telegram_token
        self.target_shape = target_shape
        self.max_upload_mb = max_upload_mb
        self.strict_model = strict_model
        self.fast_filters = fast_filters


def parse_shape(value, default=DEFAULT_TARGET_SHAPE):
    if not value:
        return default
    raw_parts = value.replace("x", ",").replace("X", ",").split(",")
    try:
        parts = tuple(int(part.strip()) for part in raw_parts if part.strip())
    except ValueError as exc:
        raise ValueError(f"Некорректный размер ROCKBOT_TARGET_SHAPE={value!r}") from exc
    if len(parts) != 3 or any(dim <= 0 for dim in parts):
        raise ValueError(f"Размер должен состоять из трех положительных чисел: {value!r}")
    return parts


def parse_filter_list(value):
    if not value:
        return DEFAULT_FAST_FILTERS
    filters = tuple(part.strip() for part in value.split(",") if part.strip())
    return filters or DEFAULT_FAST_FILTERS


def load_dotenv(path=None):
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_settings(target_shape_override=None):
    load_dotenv()
    target_shape = parse_shape(target_shape_override or os.getenv("ROCKBOT_TARGET_SHAPE"))
    max_upload_mb = int(os.getenv("ROCKBOT_MAX_UPLOAD_MB", "64"))
    strict_model = os.getenv("ROCKBOT_STRICT_MODEL", "0").lower() in {"1", "true", "yes", "on"}
    fast_filters = parse_filter_list(os.getenv("ROCKBOT_FAST_FILTERS"))
    return Settings(
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        target_shape=target_shape,
        max_upload_mb=max_upload_mb,
        strict_model=strict_model,
        fast_filters=fast_filters,
    )


def default_model_specs():
    return {
        "real": ModelSpec(
            key="real",
            title="Real",
            description="Ансамбль, обученный только на реальных образцах DRP.",
            artifact_dir=MODELS_DIR / "real",
            norm_stats=dict(REAL_NORM_STATS),
            demo_scale=4.0e-10,
            metrics={
                "dataset": "real",
                "status": "ожидает веса и production-метрики",
            },
            max_points=3000,
        ),
        "synthetic": ModelSpec(
            key="synthetic",
            title="Synthetic fine-tuned",
            description="Модель real, дообученная на синтетических структурах.",
            artifact_dir=MODELS_DIR / "synthetic",
            norm_stats=dict(SYNTHETIC_NORM_STATS),
            demo_scale=8.0e-10,
            metrics={
                "dataset": "synthetic",
                "status": "ожидает веса и production-метрики",
            },
            max_points=2000,
        ),
        "mixed": ModelSpec(
            key="mixed",
            title="Auto real/synth",
            description="Авто-выбор real или synthetic через classifier_real_vs_synth.",
            artifact_dir=MODELS_DIR / "mixed",
            norm_stats=dict(MIXED_NORM_STATS),
            demo_scale=6.0e-10,
            metrics={
                "dataset": "real + synthetic",
                "status": "ожидает веса и production-метрики",
            },
            max_points=3000,
        ),
    }


def load_model_specs():
    specs = default_model_specs()
    for key, spec in list(specs.items()):
        metadata_path = spec.artifact_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        specs[key] = ModelSpec(
            key=spec.key,
            title=metadata.get("title", spec.title),
            description=metadata.get("description", spec.description),
            artifact_dir=spec.artifact_dir,
            norm_stats=metadata.get("norm_stats", spec.norm_stats),
            demo_scale=spec.demo_scale,
            metrics=metadata.get("metrics", spec.metrics),
            max_points=int(metadata.get("max_points", spec.max_points)),
        )
    return specs


def ensure_artifact_dirs():
    for path in (ARTIFACTS_DIR, MODELS_DIR, RUNS_DIR, UPLOADS_DIR, PROJECT_VISUALS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for key in ("real", "synthetic", "mixed"):
        (MODELS_DIR / key).mkdir(parents=True, exist_ok=True)
