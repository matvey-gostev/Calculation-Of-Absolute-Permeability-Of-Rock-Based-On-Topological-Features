from collections import OrderedDict

import numpy as np


def apply_height_filtration_one(data, normal=(0, 0, 1), d=0):
    shape = data.shape
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    z_c = z - (shape[0] - 1) / 2.0
    y_c = y - (shape[1] - 1) / 2.0
    x_c = x - (shape[2] - 1) / 2.0
    n = np.asarray(normal, dtype=np.float32)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("normal must be non-zero")
    n = n / norm
    dist_from_plane = x_c * n[0] + y_c * n[1] + z_c * n[2] - d
    filtration_values = np.abs(dist_from_plane).astype(np.float32, copy=False)
    max_value = float(filtration_values.max())
    return np.where(data == 1, filtration_values, max_value).astype(np.float32, copy=False)


def apply_radial_filtration_one(data, p=None):
    shape = data.shape
    if p is None:
        p = (shape[2] // 2, shape[1] // 2, shape[0] // 2)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist_sq = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2
    filtration_values = np.sqrt(dist_sq).astype(np.float32, copy=False)
    max_value = float(filtration_values.max())
    return np.where(data == 1, filtration_values, max_value).astype(np.float32, copy=False)


def apply_line_filtration_one(
    data,
    p1=(0, 0, 0),
    p2=(255, 255, 255),
):
    shape = data.shape
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    p1_arr = np.asarray(p1, dtype=np.float32)
    p2_arr = np.asarray(p2, dtype=np.float32)
    dx, dy, dz = x - p1_arr[0], y - p1_arr[1], z - p1_arr[2]
    v = p2_arr - p1_arr
    v_norm_sq = float(np.sum(v**2) + 1e-9)
    cross_x = dy * v[2] - dz * v[1]
    cross_y = dz * v[0] - dx * v[2]
    cross_z = dx * v[1] - dy * v[0]
    filtration_values = np.sqrt((cross_x**2 + cross_y**2 + cross_z**2) / v_norm_sq).astype(np.float32, copy=False)
    max_value = float(filtration_values.max())
    return np.where(data == 1, filtration_values, max_value).astype(np.float32, copy=False)


def build_radial_filtrations():
    filtrations = OrderedDict()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                name = f"radial_{i}_{j}_{k}"
                filtrations[name] = (
                    lambda data, i=i, j=j, k=k: apply_radial_filtration_one(
                        data,
                        p=(
                            data.shape[2] // 4 * (i + 1),
                            data.shape[1] // 4 * (j + 1),
                            data.shape[0] // 4 * (k + 1),
                        ),
                    )
                )
    return filtrations


def build_line_filtrations():
    filtrations = OrderedDict()
    axial_params = [
        ("X1", 0.5, 0.25, 0.25),
        ("X2", 0.5, 0.25, 0.75),
        ("X3", 0.5, 0.75, 0.25),
        ("X4", 0.5, 0.75, 0.75),
        ("Y1", 0.25, 0.5, 0.25),
        ("Y2", 0.25, 0.5, 0.75),
        ("Y3", 0.75, 0.5, 0.25),
        ("Y4", 0.75, 0.5, 0.75),
        ("Z1", 0.25, 0.25, 0.5),
        ("Z2", 0.25, 0.75, 0.5),
        ("Z3", 0.75, 0.25, 0.5),
        ("Z4", 0.75, 0.75, 0.5),
    ]
    for name, rx, ry, rz in axial_params:
        if "X" in name:
            v = (1, 0, 0)
        elif "Y" in name:
            v = (0, 1, 0)
        else:
            v = (0, 0, 1)
        filtrations[f"line_{name}"] = (
            lambda data, rx=rx, ry=ry, rz=rz, v=v: apply_line_filtration_one(
                data,
                p1=(
                    int((data.shape[2] - 1) * rx),
                    int((data.shape[1] - 1) * ry),
                    int((data.shape[0] - 1) * rz),
                ),
                p2=(
                    int((data.shape[2] - 1) * rx) + v[0],
                    int((data.shape[1] - 1) * ry) + v[1],
                    int((data.shape[0] - 1) * rz) + v[2],
                ),
            )
        )

    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                filtrations[f"line_diag1_{i}_{j}_{k}"] = (
                    lambda data, i=i, j=j, k=k: apply_line_filtration_one(
                        data,
                        p1=(
                            (data.shape[2] - 1) // 2 + (data.shape[2] - 1) // 4 * i,
                            (data.shape[1] - 1) // 2 + (data.shape[1] - 1) // 4 * j,
                            (data.shape[0] - 1) // 2,
                        ),
                        p2=(
                            (data.shape[2] - 1) // 2,
                            (data.shape[1] - 1) // 2,
                            (data.shape[0] - 1) // 2 + (data.shape[0] - 1) // 4 * k,
                        ),
                    )
                )
                filtrations[f"line_diag2_{i}_{j}_{k}"] = (
                    lambda data, i=i, j=j, k=k: apply_line_filtration_one(
                        data,
                        p1=(
                            (data.shape[2] - 1) // 2 + (data.shape[2] - 1) // 4 * i,
                            (data.shape[1] - 1) // 2,
                            (data.shape[0] - 1) // 2 + (data.shape[0] - 1) // 4 * k,
                        ),
                        p2=(
                            (data.shape[2] - 1) // 2,
                            (data.shape[1] - 1) // 2 + (data.shape[1] - 1) // 4 * j,
                            (data.shape[0] - 1) // 2,
                        ),
                    )
                )
                filtrations[f"line_diag3_{i}_{j}_{k}"] = (
                    lambda data, i=i, j=j, k=k: apply_line_filtration_one(
                        data,
                        p1=(
                            (data.shape[2] - 1) // 2,
                            (data.shape[1] - 1) // 2 + (data.shape[1] - 1) // 4 * j,
                            (data.shape[0] - 1) // 2 + (data.shape[0] - 1) // 4 * k,
                        ),
                        p2=(
                            (data.shape[2] - 1) // 2 + (data.shape[2] - 1) // 4 * i,
                            (data.shape[1] - 1) // 2,
                            (data.shape[0] - 1) // 2,
                        ),
                    )
                )
    return filtrations


def build_height_filtrations():
    filtrations = OrderedDict()

    def plane_distance(data, axis, fraction):
        axis_index = {"x": 2, "y": 1, "z": 0}[axis]
        center = (data.shape[axis_index] - 1) / 2.0
        return data.shape[axis_index] * fraction - center

    axial_planes = [
        ("height_x1", (1, 0, 0), "x", 0.25),
        ("height_x2", (1, 0, 0), "x", 0.75),
        ("height_y1", (0, 1, 0), "y", 0.25),
        ("height_y2", (0, 1, 0), "y", 0.75),
        ("height_z1", (0, 0, 1), "z", 0.25),
        ("height_z2", (0, 0, 1), "z", 0.75),
    ]
    for name, normal, axis, fraction in axial_planes:
        filtrations[name] = (
            lambda data, normal=normal, axis=axis, fraction=fraction: apply_height_filtration_one(
                data, normal=normal, d=plane_distance(data, axis, fraction)
            )
        )

    for s_x in (-1, 1):
        for s_y in (-1, 1):
            for s_z in (-1, 1):
                normal = (s_y * s_z, s_x * s_z, s_x * s_y)
                signs = f"{'+' if s_x > 0 else '-'}{'+' if s_y > 0 else '-'}{'+' if s_z > 0 else '-'}"
                name = f"height_diag_{signs}"
                filtrations[name] = (
                    lambda data, normal=normal, s_x=s_x, s_y=s_y, s_z=s_z: apply_height_filtration_one(
                        data,
                        normal=normal,
                        d=s_x * s_y * s_z * (min(data.shape) // 4),
                    )
                )
    return filtrations


def build_filtrations(include_gtda=False):
    filtrations = OrderedDict()
    filtrations.update(build_radial_filtrations())
    filtrations.update(build_line_filtrations())
    filtrations.update(build_height_filtrations())
    if include_gtda:
        filtrations.update(build_gtda_filtrations())
    return filtrations


def build_gtda_filtrations():
    try:
        from gtda.images import DensityFiltration, DilationFiltration, ErosionFiltration, SignedDistanceFiltration
    except ImportError:
        return OrderedDict()

    def apply_gtda_filtration(data, filtration):
        filtered = filtration.fit_transform(data[np.newaxis, ...])[0]
        max_value = float(filtered.max())
        return np.where(data == 1, filtered, max_value).astype(np.float32, copy=False)

    return OrderedDict(
        {
            "density": lambda data: apply_gtda_filtration(data, DensityFiltration()),
            "dilation": lambda data: apply_gtda_filtration(data, DilationFiltration()),
            "erosion": lambda data: apply_gtda_filtration(data, ErosionFiltration()),
            "signed_distance": lambda data: apply_gtda_filtration(data, SignedDistanceFiltration()),
        }
    )


ALL_FILTRATIONS = build_filtrations(include_gtda=False)
