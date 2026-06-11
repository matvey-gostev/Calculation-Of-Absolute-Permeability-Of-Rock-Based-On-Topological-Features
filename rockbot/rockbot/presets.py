import numpy as np
from scipy import ndimage


class Preset:
    def __init__(self, key, title, description):
        self.key = key
        self.title = title
        self.description = description


PRESETS = (
    Preset("sandstone_channels", "Sandstone channels", "Связанная пористая сеть после сглаженного шума."),
    Preset("carbonate_vugs", "Carbonate vugs", "Редкая матрица пор и несколько крупных пустот."),
    Preset("fractured_granite", "Fractured granite", "Плотная порода с пересекающимися трещинами."),
    Preset("laminated_shale", "Laminated shale", "Тонкие горизонтальные прослои с низкой пористостью."),
    Preset("bead_pack", "Bead pack", "Синтетическая среда между сферическими зернами."),
)


def preset_by_key(key):
    return next((preset for preset in PRESETS if preset.key == key), None)


def generate_preset(key, shape):
    if key == "sandstone_channels":
        return _sandstone_channels(shape)
    if key == "carbonate_vugs":
        return _carbonate_vugs(shape)
    if key == "fractured_granite":
        return _fractured_granite(shape)
    if key == "laminated_shale":
        return _laminated_shale(shape)
    if key == "bead_pack":
        return _bead_pack(shape)
    raise KeyError(f"Unknown preset: {key}")


def _sandstone_channels(shape):
    rng = np.random.default_rng(2026)
    noise = rng.random(shape, dtype=np.float32)
    sigma = max(1.5, min(shape) / 22.0)
    smooth = ndimage.gaussian_filter(noise, sigma=sigma)
    threshold = float(np.quantile(smooth, 0.68))
    cube = smooth > threshold
    cube = ndimage.binary_closing(cube, iterations=max(1, min(shape) // 96))
    return cube.astype(np.uint8)


def _carbonate_vugs(shape):
    rng = np.random.default_rng(1703)
    base = rng.random(shape, dtype=np.float32)
    smooth = ndimage.gaussian_filter(base, sigma=max(1.0, min(shape) / 30.0))
    cube = smooth > np.quantile(smooth, 0.84)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for _ in range(10):
        center = (
            rng.integers(shape[0] // 8, max(shape[0] // 8 + 1, shape[0] * 7 // 8)),
            rng.integers(shape[1] // 8, max(shape[1] // 8 + 1, shape[1] * 7 // 8)),
            rng.integers(shape[2] // 8, max(shape[2] // 8 + 1, shape[2] * 7 // 8)),
        )
        radius = rng.uniform(min(shape) * 0.035, min(shape) * 0.095)
        sphere = (
            ((z - center[0]) / max(radius * 0.9, 1)) ** 2
            + ((y - center[1]) / max(radius * 1.1, 1)) ** 2
            + ((x - center[2]) / max(radius, 1)) ** 2
            <= 1.0
        )
        cube |= sphere
    return cube.astype(np.uint8)


def _fractured_granite(shape):
    rng = np.random.default_rng(414)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    coords = [
        x / max(shape[2] - 1, 1),
        y / max(shape[1] - 1, 1),
        z / max(shape[0] - 1, 1),
    ]
    cube = np.zeros(shape, dtype=bool)
    for _ in range(8):
        normal = rng.normal(size=3)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        offset = rng.uniform(-0.35, 0.35)
        thickness = rng.uniform(0.005, 0.018)
        plane = normal[0] * (coords[0] - 0.5) + normal[1] * (coords[1] - 0.5) + normal[2] * (coords[2] - 0.5)
        cube |= np.abs(plane - offset) < thickness
    cube |= rng.random(shape) < 0.012
    cube = ndimage.binary_dilation(cube, iterations=max(1, min(shape) // 160))
    return cube.astype(np.uint8)


def _laminated_shale(shape):
    rng = np.random.default_rng(90210)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    periodic = (np.sin(z / max(shape[0], 1) * np.pi * 15) + 1.0) / 2.0
    layer_probability = 0.015 + 0.13 * (periodic > 0.83)
    noise = rng.random(shape, dtype=np.float32)
    cube = noise < layer_probability
    cube = ndimage.binary_opening(cube, iterations=1)
    cube = ndimage.binary_dilation(cube, structure=np.ones((1, 2, 2), dtype=bool), iterations=1)
    return cube.astype(np.uint8)


def _bead_pack(shape):
    rng = np.random.default_rng(77)
    pores = np.ones(shape, dtype=bool)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    grain_count = max(14, min(54, int(np.prod(shape) / (96**3) * 20)))
    for _ in range(grain_count):
        center = (
            rng.integers(0, shape[0]),
            rng.integers(0, shape[1]),
            rng.integers(0, shape[2]),
        )
        radius = rng.uniform(min(shape) * 0.055, min(shape) * 0.12)
        solid = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2
        pores &= ~solid
    pores = ndimage.binary_opening(pores, iterations=1)
    return pores.astype(np.uint8)
