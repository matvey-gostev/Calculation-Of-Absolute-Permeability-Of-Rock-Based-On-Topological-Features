import math
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import CLASSIFIER_PATH, RUNS_DIR, load_model_specs
from .filtrations import ALL_FILTRATIONS
from .models import ModelLoadError, TorchEnsemble, load_domain_classifier, normalize_pd_points
from .preprocessing import compute_morphology_features, validate_and_prepare_volume
from .visualization import render_visualizations


class InferenceResult:
    def __init__(
        self,
        sample_name,
        model_key,
        model_title,
        permeability,
        log_permeability,
        mode,
        used_filters,
        skipped_filters,
        validation,
        features,
        image_paths=None,
        warnings=None,
        elapsed_sec=0.0,
        model_note=None,
    ):
        self.sample_name = sample_name
        self.model_key = model_key
        self.model_title = model_title
        self.permeability = permeability
        self.log_permeability = log_permeability
        self.mode = mode
        self.used_filters = used_filters
        self.skipped_filters = skipped_filters
        self.validation = validation
        self.features = features
        self.image_paths = image_paths or []
        self.warnings = warnings or []
        self.elapsed_sec = elapsed_sec
        self.model_note = model_note

    def result_text(self):
        mode_text = "модельный ансамбль" if self.mode == "model" else "demo-оценка"
        lines = [
            f"Готово: {self.sample_name}",
            f"Модель: {self.model_title} ({self.model_key})",
        ]
        if self.model_note:
            lines.append(self.model_note)
        lines.extend(
            [
                f"Режим: {mode_text}",
                f"log(LBM): {self.log_permeability:.4f}",
                f"LBM: {self.permeability:.4e}",
                "",
                f"Пористость: {self.features['porosity']:.3f}",
                f"Плотность границ: {self.features['surface_density']:.3f}",
                f"Предобработка: {self.validation.compact_text()}",
                f"Фильтрации: {len(self.used_filters)} использовано, {len(self.skipped_filters)} пропущено",
                f"Время: {self.elapsed_sec:.1f} c",
            ]
        )
        all_warnings = [*self.validation.warnings, *self.warnings]
        if all_warnings:
            lines.append("")
            lines.append("Замечания:")
            lines.extend(f"- {warning}" for warning in all_warnings[:6])
        return "\n".join(lines)


class PermeabilityEngine:
    def __init__(self, settings):
        self.settings = settings
        self.specs = load_model_specs()
        self._ensembles = {}
        self._domain_classifier = False

    def infer(self, volume, model_key="mixed", sample_name="sample"):
        started = time.monotonic()
        if model_key not in self.specs:
            model_key = "mixed"
        spec = self.specs[model_key]
        max_voxels = self._max_voxels()
        cube, validation = validate_and_prepare_volume(volume, self.settings.target_shape, max_voxels=max_voxels)
        features = compute_morphology_features(cube)

        warnings = []
        used_filters = []
        skipped_filters = []
        pd_for_plot = None
        model_note = None

        if spec.key == "mixed":
            try:
                log_predictions, used_filters, skipped_filters, pd_for_plot, selected_spec = self._predict_auto_with_topology(cube)
                if log_predictions:
                    log_perm = float(np.mean(log_predictions))
                    mode = "model"
                    model_note = f"Auto domain: {selected_spec.title} ({selected_spec.key})"
                else:
                    raise RuntimeError("Не удалось получить ни одного прогноза для auto real/synth.")
            except Exception as exc:
                if self.settings.strict_model:
                    raise
                warnings.append(f"Модельный инференс недоступен, включен demo fallback: {exc}")
                log_perm = heuristic_log_permeability(features, spec.demo_scale)
                mode = "demo"
        else:
            ensemble = self._get_ensemble(spec.key)
            if ensemble is not None:
                try:
                    log_predictions, used_filters, skipped_filters, pd_for_plot = self._predict_with_topology(cube, ensemble)
                    if log_predictions:
                        log_perm = float(np.mean(log_predictions))
                        mode = "model"
                    else:
                        raise RuntimeError("Не удалось получить ни одной PD-точки для выбранных фильтраций.")
                except Exception as exc:
                    if self.settings.strict_model:
                        raise
                    warnings.append(f"Модельный инференс недоступен, включен demo fallback: {exc}")
                    log_perm = heuristic_log_permeability(features, spec.demo_scale)
                    mode = "demo"
            else:
                if self.settings.strict_model:
                    raise ModelLoadError(f"Для модели {spec.key} не загружены веса.")
                warnings.append("Веса модели не найдены: результат рассчитан demo-оценщиком, не production-моделью.")
                log_perm = heuristic_log_permeability(features, spec.demo_scale)
                mode = "demo"

        permeability = float(math.exp(log_perm)) if math.isfinite(log_perm) else float("nan")
        run_dir = RUNS_DIR / f"{int(time.time())}_{uuid4().hex[:8]}"
        image_paths = render_visualizations(cube, run_dir, sample_name, features, pd_for_plot)
        return InferenceResult(
            sample_name=sample_name,
            model_key=spec.key,
            model_title=spec.title,
            permeability=permeability,
            log_permeability=log_perm,
            mode=mode,
            used_filters=used_filters,
            skipped_filters=skipped_filters,
            validation=validation,
            features=features,
            image_paths=image_paths,
            warnings=warnings,
            elapsed_sec=time.monotonic() - started,
            model_note=model_note,
        )

    def _max_voxels(self):
        bytes_limit = self.settings.max_upload_mb * 1024 * 1024
        return max(1, bytes_limit // 4)

    def _get_ensemble(self, model_key):
        if model_key in self._ensembles:
            return self._ensembles[model_key]
        spec = self.specs[model_key]
        try:
            ensemble = TorchEnsemble.load(spec)
        except Exception:
            ensemble = None
        self._ensembles[model_key] = ensemble
        return ensemble

    def _predict_with_topology(self, cube, ensemble):
        pd_by_filter, skipped_filters = self._compute_pd_by_filter(cube)
        log_predictions, used_filters, pd_for_plot = self._predict_from_pd_points(pd_by_filter, ensemble)
        skipped_filters.extend(name for name, points in pd_by_filter if points.size and name not in used_filters)
        return log_predictions, used_filters, skipped_filters, pd_for_plot

    def _predict_auto_with_topology(self, cube):
        pd_by_filter, skipped_filters = self._compute_pd_by_filter(cube)
        if not pd_by_filter:
            raise RuntimeError("Не удалось получить PD-точки для auto real/synth.")
        classifier = self._get_domain_classifier()
        if classifier is None:
            raise RuntimeError(f"Не найден или не загружен классификатор {CLASSIFIER_PATH}.")
        domain = self._classify_domain(pd_by_filter, classifier)
        selected_key = "real" if int(domain) == 0 else "synthetic"
        selected_spec = self.specs[selected_key]
        ensemble = self._get_ensemble(selected_key)
        if ensemble is None:
            raise ModelLoadError(f"Для выбранной модели {selected_key} не загружены веса.")
        log_predictions, used_filters, pd_for_plot = self._predict_from_pd_points(pd_by_filter, ensemble)
        skipped_filters.extend(name for name, points in pd_by_filter if points.size and name not in used_filters)
        return log_predictions, used_filters, skipped_filters, pd_for_plot, selected_spec

    def _compute_pd_by_filter(self, cube):
        try:
            compute_ph = load_cripser_compute_ph()
        except ImportError as exc:
            raise RuntimeError("не установлен cripser") from exc

        selected_filters = self._selected_filters()
        pd_by_filter = []
        skipped_filters = []
        for filt_name in selected_filters:
            filt = ALL_FILTRATIONS.get(filt_name)
            if filt is None:
                skipped_filters.append(filt_name)
                continue
            grayscale = filt(cube)
            raw = compute_ph(grayscale, maxdim=2)
            if raw is None or len(raw) == 0:
                skipped_filters.append(filt_name)
                continue
            birth_death_dim = raw[:, [1, 2, 0]].copy()
            birth_death_dim[birth_death_dim[:, 1] > 1e300, 1] = np.inf
            points = birth_death_dim[:, [2, 0, 1]]
            pd_by_filter.append((filt_name, points))
        return pd_by_filter, skipped_filters

    def _predict_from_pd_points(self, pd_by_filter, ensemble):
        log_predictions = []
        used_filters = []
        pd_for_plot = None
        for filt_name, points in pd_by_filter:
            normalized = normalize_pd_points(points, ensemble.spec.norm_stats, max_points=ensemble.max_points)
            if normalized is None:
                continue
            log_predictions.append(ensemble.predict_log(normalized))
            used_filters.append(filt_name)
            if pd_for_plot is None:
                pd_for_plot = points
        return log_predictions, used_filters, pd_for_plot

    def _get_domain_classifier(self):
        if self._domain_classifier is not False:
            return self._domain_classifier
        try:
            self._domain_classifier = load_domain_classifier(CLASSIFIER_PATH)
        except Exception:
            self._domain_classifier = None
        return self._domain_classifier

    def _classify_domain(self, pd_by_filter, classifier):
        diagrams = [points for _, points in pd_by_filter if points.size]
        h0 = np.median([(points[:, 0] == 0).sum() for points in diagrams])
        h1 = np.median([(points[:, 0] == 1).sum() for points in diagrams])
        persistences = [np.nan_to_num(points[:, 2] - points[:, 1], nan=0.0, posinf=0.0, neginf=0.0) for points in diagrams]
        pers_mean = np.median([values.mean() if values.size else 0.0 for values in persistences])
        pers_std = np.median([values.std() if values.size else 0.0 for values in persistences])
        features = np.array([[h0, h1, pers_mean, pers_std]], dtype=np.float32)
        return int(classifier.predict(features)[0])

    def _selected_filters(self):
        if self.settings.fast_filters:
            return self.settings.fast_filters
        return tuple(ALL_FILTRATIONS.keys())


def heuristic_log_permeability(features, scale):
    porosity = min(max(features["porosity"], 1e-4), 0.98)
    surface = max(features["surface_density"], 1e-3)
    edge_contact = max(features["edge_contact"], 1e-4)
    anisotropy = features["anisotropy"]
    kozeny_proxy = (porosity**3) / ((1.0 - porosity + 0.02) ** 2 * (surface + 0.015) ** 2)
    connectivity_boost = 0.75 + 0.55 * edge_contact + 0.25 * min(anisotropy, 1.0)
    permeability = scale * kozeny_proxy * connectivity_boost
    permeability = float(np.clip(permeability, 1e-18, 1e-7))
    return float(math.log(permeability))


def load_cripser_compute_ph():
    try:
        import cripser

        return cripser.computePH
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        if not patch_cripser_py39_pathlike():
            raise
        for name in list(sys.modules):
            if name == "cripser" or name.startswith("cripser."):
                sys.modules.pop(name, None)
        import cripser

        return cripser.computePH


def patch_cripser_py39_pathlike():
    for base in sys.path:
        if not base:
            continue
        path = Path(base) / "cripser" / "image_loader.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "PathLike = str | Path" not in text:
            return True
        text = text.replace("from typing import Any, Sequence", "from typing import Any, Sequence, Union")
        text = text.replace("PathLike = str | Path", "PathLike = Union[str, Path]")
        path.write_text(text, encoding="utf-8")
        return True
    return False
