import json

import numpy as np
from scipy import ndimage


class VolumeValidationError(ValueError):
    pass


class ValidationReport:
    def __init__(
        self,
        original_shape,
        target_shape,
        nan_count,
        inf_count,
        finite_min,
        finite_max,
        resized,
        binarization,
        warnings=None,
    ):
        self.original_shape = original_shape
        self.target_shape = target_shape
        self.nan_count = nan_count
        self.inf_count = inf_count
        self.finite_min = finite_min
        self.finite_max = finite_max
        self.resized = resized
        self.binarization = binarization
        self.warnings = warnings or []

    def compact_text(self):
        parts = [
            f"shape {self.original_shape} -> {self.target_shape}",
            f"NaN: {self.nan_count}",
            f"Inf: {self.inf_count}",
            f"binary: {self.binarization}",
        ]
        if self.resized:
            parts.append("resized")
        return "; ".join(parts)


def load_volume_file(path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        keys = list(data.keys())
        if not keys:
            raise VolumeValidationError("В .npz нет массивов.")
        preferred = next((key for key in ("volume", "cube", "data", "arr_0") if key in data), keys[0])
        return data[preferred]
    if suffix in {".json", ".jsn"}:
        return np.asarray(json.loads(path.read_text(encoding="utf-8")))
    if suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    if suffix == ".mat":
        try:
            import h5py
        except ImportError as exc:
            raise VolumeValidationError("Для .mat нужен пакет h5py.") from exc
        with h5py.File(path, "r") as h5_file:
            dataset = _find_first_3d_h5_dataset(h5_file)
            if dataset is None:
                raise VolumeValidationError("В .mat не найден 3D dataset.")
            return np.asarray(dataset)
    raise VolumeValidationError("Поддерживаются .npy, .npz, .json, .mat и простые .txt/.csv с 3D-данными.")


def _find_first_3d_h5_dataset(h5_file):
    found = None

    def visitor(_name, obj):
        nonlocal found
        if found is None and getattr(obj, "ndim", None) == 3:
            found = obj

    h5_file.visititems(visitor)
    return found


def validate_and_prepare_volume(
    volume,
    target_shape,
    max_voxels=None,
):
    arr = np.asarray(volume)
    arr = np.squeeze(arr)
    original_shape = tuple(int(dim) for dim in arr.shape)
    if arr.ndim != 3:
        raise VolumeValidationError(f"Нужен 3D-массив, получено {arr.ndim}D с shape={original_shape}.")
    if any(dim < 2 for dim in arr.shape):
        raise VolumeValidationError(f"Каждая ось должна иметь хотя бы 2 элемента, получено shape={original_shape}.")
    if max_voxels is not None and arr.size > max_voxels:
        raise VolumeValidationError(
            f"Массив слишком большой: {arr.size:,} voxels, лимит {max_voxels:,}."
        )

    arr = arr.astype(np.float32, copy=False)
    finite_mask = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    warnings = []
    if not finite_mask.any():
        raise VolumeValidationError("Во входном массиве нет ни одного конечного значения.")

    finite_values = arr[finite_mask]
    finite_min = float(finite_values.min())
    finite_max = float(finite_values.max())
    was_binary = _looks_binary(finite_values)

    if nan_count or inf_count:
        arr = fill_missing_nearest(arr)
        warnings.append("Пропуски и inf заполнены ближайшими конечными значениями.")

    resized = tuple(arr.shape) != tuple(target_shape)
    if resized:
        order = 0 if was_binary else 1
        arr = resize_to_shape(arr, target_shape, order=order)
        warnings.append("Массив интерполирован до размера модели.")

    binary, binarization = binarize_volume(arr, was_binary=was_binary)
    report = ValidationReport(
        original_shape=original_shape,
        target_shape=tuple(target_shape),
        nan_count=nan_count,
        inf_count=inf_count,
        finite_min=finite_min,
        finite_max=finite_max,
        resized=resized,
        binarization=binarization,
        warnings=warnings,
    )
    return binary.astype(np.uint8, copy=False), report


def _looks_binary(values):
    if values.size == 0:
        return False
    sample = values if values.size <= 1_000_000 else values[:: max(1, values.size // 1_000_000)]
    unique = np.unique(sample)
    return len(unique) <= 2 and np.all(np.isin(unique, [0, 1, False, True]))


def fill_missing_nearest(arr):
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if not valid.any():
        raise VolumeValidationError("Нельзя интерполировать массив без конечных значений.")
    indices = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    filled = arr[tuple(indices)]
    return filled.astype(np.float32, copy=False)


def resize_to_shape(arr, target_shape, order=1):
    zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
    resized = ndimage.zoom(arr, zoom=zoom_factors, order=order)
    if resized.shape != tuple(target_shape):
        padded = np.zeros(target_shape, dtype=resized.dtype)
        target_slices = tuple(slice(0, min(resized.shape[i], target_shape[i])) for i in range(3))
        source_slices = tuple(slice(0, min(resized.shape[i], target_shape[i])) for i in range(3))
        padded[target_slices] = resized[source_slices]
        resized = padded
    return resized.astype(np.float32, copy=False)


def binarize_volume(arr, was_binary=False):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise VolumeValidationError("После интерполяции не осталось конечных значений.")
    if was_binary or _looks_binary(finite):
        unique = np.unique(finite)
        if len(unique) == 1:
            value = int(unique[0] > 0)
            return np.full(arr.shape, value, dtype=np.uint8), "constant-binary"
        threshold = float(unique.min() + (unique.max() - unique.min()) / 2.0)
        return (arr > threshold).astype(np.uint8), f"binary-threshold={threshold:.4g}"

    finite_min = float(finite.min())
    finite_max = float(finite.max())
    if 0.0 <= finite_min and finite_max <= 1.0:
        threshold = 0.5
        method = "range-0-1-threshold=0.5"
    else:
        threshold = float(np.nanmedian(finite))
        method = f"median-threshold={threshold:.4g}"
    return (arr > threshold).astype(np.uint8), method


def compute_morphology_features(cube):
    cube_f = cube.astype(np.float32, copy=False)
    porosity = float(cube_f.mean())
    diff_x = np.abs(np.diff(cube_f, axis=2)).mean() if cube.shape[2] > 1 else 0.0
    diff_y = np.abs(np.diff(cube_f, axis=1)).mean() if cube.shape[1] > 1 else 0.0
    diff_z = np.abs(np.diff(cube_f, axis=0)).mean() if cube.shape[0] > 1 else 0.0
    surface_density = float((diff_x + diff_y + diff_z) / 3.0)
    axis_profiles = np.array(
        [
            cube_f.mean(axis=(1, 2)).std(),
            cube_f.mean(axis=(0, 2)).std(),
            cube_f.mean(axis=(0, 1)).std(),
        ],
        dtype=np.float32,
    )
    anisotropy = float(axis_profiles.max() - axis_profiles.min())
    edge_contact = float(
        np.mean(
            [
                cube_f[0, :, :].mean(),
                cube_f[-1, :, :].mean(),
                cube_f[:, 0, :].mean(),
                cube_f[:, -1, :].mean(),
                cube_f[:, :, 0].mean(),
                cube_f[:, :, -1].mean(),
            ]
        )
    )
    return {
        "porosity": porosity,
        "surface_density": surface_density,
        "anisotropy": anisotropy,
        "edge_contact": edge_contact,
    }
