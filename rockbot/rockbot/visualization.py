import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rockbot-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .filtrations import apply_radial_filtration_one


BINARY_CMAP = ListedColormap(["#6a6863", "white"])


def render_visualizations(
    cube,
    out_dir,
    sample_name,
    features,
    pd_points=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        render_pore_space(cube, out_dir / "pore_space.png", sample_name, features),
        render_filtration_overview(cube, out_dir / "filtration.png", sample_name),
    ]
    if pd_points is not None and len(pd_points):
        paths.append(render_pd_points(pd_points, out_dir / "persistent_diagram.png", sample_name))
    return paths


def render_pore_space(cube, path, sample_name, features):
    fig = plt.figure(figsize=(10, 6), dpi=160)
    grid = fig.add_gridspec(2, 3)
    ax_cube = fig.add_subplot(grid[:, :2])
    ax_profile = fig.add_subplot(grid[0, 2])
    ax_hist = fig.add_subplot(grid[1, 2])

    cube_image = render_project_cube_image(cube, sample_name)
    if cube_image is not None:
        ax_cube.imshow(cube_image)
    ax_cube.set_title("Поровое пространство", fontsize=11)
    ax_cube.axis("off")

    profiles = [
        cube.mean(axis=(1, 2)),
        cube.mean(axis=(0, 2)),
        cube.mean(axis=(0, 1)),
    ]
    for label, profile in zip(("z", "y", "x"), profiles):
        ax_profile.plot(profile, label=label, linewidth=1.2)
    ax_profile.set_title("Пористость по осям", fontsize=10)
    ax_profile.set_ylim(0, 1)
    ax_profile.grid(alpha=0.25)
    ax_profile.legend(fontsize=8)

    values = [features["porosity"], 1.0 - features["porosity"]]
    ax_hist.barh([0, 1], values, color=["#f5f5f5", "#6a6863"], edgecolor="#2f2f2f", linewidth=0.7)
    ax_hist.set_yticks([0, 1], ["Поры", "Руда"], fontsize=9)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Доля объема", fontsize=9)
    ax_hist.set_title("Морфология", fontsize=11)
    ax_hist.invert_yaxis()
    ax_hist.grid(axis="x", alpha=0.2)
    for idx, value in enumerate(values):
        ax_hist.text(min(value + 0.025, 0.98), idx, f"{value:.2f}", va="center", fontsize=8)

    fig.suptitle(sample_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_project_cube_image(cube, sample_name):
    return render_project_surface_image(cube, sample_name=sample_name)


def render_filtration_overview(cube, path, sample_name):
    center = (cube.shape[2] // 2, cube.shape[1] // 2, cube.shape[0] // 2)
    filtration = apply_radial_filtration_one(cube, center)
    finite = filtration[np.isfinite(filtration)]
    if finite.size:
        filtration = (filtration - finite.min()) / (finite.max() - finite.min() + 1e-8)

    rock_img = render_project_cube_image(cube, sample_name)
    colored_rock_img = render_project_colored_rock_image(cube, filtration)

    axis = "z"
    max_k = max(cube.shape[1] - 1, 0)
    middle_k = max_k // 2
    vis_pore = np.where(cube == 1, filtration, -0.1)
    pore_cmap = pore_filtration_cmap()

    fig = plt.figure(figsize=(15, 9), dpi=145)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.18])
    fig.text(0.06, 0.96, f"Образец: {sample_name}_radial", fontsize=17, va="center")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    cax = fig.add_axes([0.42, 0.955, 0.50, 0.014])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Filtration values", labelpad=-54, fontsize=13)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["min", "max"], fontsize=12)

    top_axes = (fig.add_subplot(grid[0, 0:3]), fig.add_subplot(grid[0, 3:6]))
    for ax, image in zip(top_axes, (rock_img, colored_rock_img)):
        if image is not None:
            ax.imshow(image)
        ax.axis("off")

    ax_bin = fig.add_subplot(grid[1, 0:2])
    ax_bin.imshow(cube[:, middle_k, :], cmap=BINARY_CMAP, interpolation="nearest")
    ax_bin.text(-0.05, 0.5, f"{axis} = {middle_k}", transform=ax_bin.transAxes, rotation="vertical", va="center", ha="right", fontsize=13)
    ax_bin.axis("off")

    ax_all = fig.add_subplot(grid[1, 2:4])
    ax_all.imshow(filtration[:, middle_k, :], cmap="jet", interpolation="nearest")
    ax_all.axis("off")

    ax_pore = fig.add_subplot(grid[1, 4:6])
    ax_pore.imshow(vis_pore[:, middle_k, :], cmap=pore_cmap, interpolation="nearest")
    ax_pore.axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.055, wspace=0.08, hspace=0.14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_project_colored_rock_image(cube, filtration):
    return render_project_surface_image(cube, filtration=filtration)


def render_project_surface_image(
    cube,
    sample_name=None,
    filtration=None,
    max_side=96,
):
    try:
        from skimage import measure
    except Exception:
        return render_project_slice_fallback(cube)

    data = _downsample_for_surface(cube, max_side=max_side).astype(np.float32, copy=False)
    if data.min() == data.max():
        return render_project_slice_fallback(cube)

    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(np.flip(data, axis=0), level=0.5, step_size=1)
    except Exception:
        return render_project_slice_fallback(cube)
    verts_xyz = verts_zyx[:, [2, 1, 0]]

    fig = plt.figure(figsize=(6.35, 6.35), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts_xyz[faces], linewidths=0.04, alpha=1.0)
    mesh.set_edgecolor((0.18, 0.18, 0.18, 0.16))
    if filtration is None:
        mesh.set_facecolor((0.38, 0.37, 0.34, 1.0))
    else:
        filt = _downsample_for_surface(filtration, max_side=max_side)
        filt = np.flip(filt, axis=0)
        vertex_values = _sample_vertex_values(filt, verts_zyx)
        face_values = vertex_values[faces].mean(axis=1)
        mesh.set_facecolor(plt.get_cmap("jet")(np.clip(face_values, 0, 1)))
    ax.add_collection3d(mesh)
    _set_project_3d_axes(ax, data.shape)
    _draw_wire_cube(ax, data.shape)
    if sample_name:
        ax.set_title(sample_name, fontsize=10, pad=0)
    image = _figure_to_rgb(fig)
    plt.close(fig)
    return image


def _downsample_for_surface(array, max_side):
    steps = tuple(max(1, int(np.ceil(dim / max_side))) for dim in array.shape)
    return array[:: steps[0], :: steps[1], :: steps[2]]


def _sample_vertex_values(values, verts):
    indices = np.rint(verts).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, values.shape[0] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, values.shape[1] - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, values.shape[2] - 1)
    return values[indices[:, 0], indices[:, 1], indices[:, 2]]


def _set_project_3d_axes(ax, shape):
    ax.set_xlim(0, shape[2] - 1)
    ax.set_ylim(0, shape[1] - 1)
    ax.set_zlim(0, shape[0] - 1)
    ax.view_init(elev=18, azim=-63)
    ax.set_box_aspect((shape[2], shape[1], shape[0]))
    ax.set_axis_off()
    ax.set_facecolor("white")


def _draw_wire_cube(ax, shape):
    z_max, y_max, x_max = (dim - 1 for dim in shape)
    corners = np.array(
        [
            [0, 0, 0],
            [x_max, 0, 0],
            [x_max, y_max, 0],
            [0, y_max, 0],
            [0, 0, z_max],
            [x_max, 0, z_max],
            [x_max, y_max, z_max],
            [0, y_max, z_max],
        ],
        dtype=float,
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        xs, ys, zs = zip(corners[start], corners[end])
        ax.plot(xs, ys, zs, color="black", linewidth=0.6, alpha=0.55)


def _figure_to_rgb(fig):
    fig.tight_layout(pad=0.02)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def pore_filtration_cmap():
    jet_colors = plt.get_cmap("jet")(np.linspace(0, 1, 256))
    colors = np.vstack([np.array([106 / 255, 104 / 255, 99 / 255, 1.0]), jet_colors])
    return LinearSegmentedColormap.from_list("jet_with_grey", colors)


def render_project_slice_fallback(cube):
    zc, yc, xc = (dim // 2 for dim in cube.shape)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), dpi=120)
    for ax, (axis, idx, image) in zip(
        axes,
        (("Z", zc, cube[zc, :, :]), ("Y", yc, cube[:, yc, :]), ("X", xc, cube[:, :, xc])),
    ):
        ax.imshow(image, cmap=BINARY_CMAP, interpolation="nearest")
        ax.set_title(f"{axis}={idx}", fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_pd_points(points, path, sample_name, min_persistence=0.01):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=165)
    colors = ["#ef553b", "#00cc96", "#ab63fa"]
    dim_names = ["H0", "H1", "H2"]
    birth = points[:, 1]
    death = points[:, 2]
    death_plot_all = np.where(np.isinf(death), np.nan, death)
    finite = np.isfinite(birth) & np.isfinite(death_plot_all)
    max_axis = float(max(birth[finite].max(), death_plot_all[finite].max())) if finite.any() else 1.0
    max_axis = max(max_axis * 1.05, 1.0)
    for dim in range(3):
        ax = axes[dim]
        mask = (points[:, 0] == dim) & finite & ((death_plot_all - birth) >= min_persistence)
        ax.scatter(
            birth[mask],
            death_plot_all[mask],
            s=10,
            alpha=0.6,
            color=colors[dim],
            edgecolors="white",
            linewidth=0.3,
        )
        ax.plot([0, max_axis], [0, max_axis], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis * 1.08)
        ax.set_title(f"{dim_names[dim]} ({int(mask.sum())})", fontsize=10)
        ax.set_xlabel("Birth", fontsize=8)
        ax.set_ylabel("Death", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{sample_name}: persistent diagram", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _downsample_for_voxels(cube, max_side):
    steps = tuple(max(1, int(np.ceil(dim / max_side))) for dim in cube.shape)
    return cube[:: steps[0], :: steps[1], :: steps[2]]
