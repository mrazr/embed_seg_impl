import typing

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, draw, util

from post_processing import Cluster, Instance


def visualize_pixel_offsets(offset_yx_map: np.ndarray, image: np.ndarray, seediness: np.ndarray, alpha=0.5) -> typing.Tuple[np.ndarray, np.ndarray]:
    angles = np.rad2deg(np.arctan2(offset_yx_map[0], offset_yx_map[1])) + 180
    norms = np.linalg.norm(offset_yx_map, axis=0)
    norms = np.where(seediness > 0.5, norms, 0.0)
    norms = norms / np.max(norms)

    hsv = np.dstack((angles / 360.0, norms, np.ones_like(norms)))

    rgb = color.hsv2rgb(hsv)

    rgb_im = util.img_as_float32(np.dstack((image,) * 3))

    overlaid = alpha * rgb + (1.0 - alpha) * rgb_im

    overlaid = np.where(np.dstack((seediness > 0.5, ) * 3), overlaid, rgb_im)

    return rgb, overlaid


def visualize_clusters(clusters: typing.List[Cluster], image: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(image, dtype=np.uint32)

    for cluster in clusters:
        for coord in cluster.pixels:
            if not (0 <= coord[0] < image.shape[0] and 0 <= coord[1] < image.shape[1]):
                continue
            coord = np.round(coord).astype(np.uint32)
            rr, cc = draw.disk(tuple(coord), 3, shape=image.shape)
            vis[rr, cc] = cluster.id

    rgb_vis = np.round(255 * color.label2rgb(vis, image)).astype(np.uint8)

    for cluster in clusters:
        center = np.round(cluster.center).astype(np.uint8)
        rr, cc = draw.disk(tuple(center), 5, shape=vis.shape)
        rgb_vis[rr, cc] = [255, 255, 255]
        rr, cc = draw.circle_perimeter(center[0], center[1], round(cluster.sigma), shape=vis.shape)
        rgb_vis[rr, cc] = [255, 255, 255]

    return rgb_vis


def visualize_instances(instances: typing.List[Instance], image: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(image, dtype=np.uint32)

    for instance in instances:
        for pixel in instance.pixels:
            vis[pixel[0], pixel[1]] = instance.id

    rgb = np.round(255 * color.label2rgb(vis, image)).astype(np.uint8)

    return rgb


def visualize_offset_vectors(image: np.ndarray, seediness: np.ndarray, offset_yx_map: np.ndarray, ax):
    yy, xx = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    # yy, xx = yy[::4, ::4], xx[::4, ::4]
    step = 20
    yys, xxs = np.nonzero(seediness > 0.5)

    arr_xxs = xx[yys, xxs]
    arr_yys = yy[yys, xxs]

    offsets_y = offset_yx_map[yys, xxs, 0]
    offsets_x = offset_yx_map[yys, xxs, 1]

    offs_angles = (np.rad2deg(np.arctan2(offset_yx_map[:, :, 0], offset_yx_map[:, :, 1])) + 180).astype(np.uint32)

    # fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    ax.imshow(image, cmap='gray')
    ax.quiver(arr_xxs[::step], arr_yys[::step], offsets_x[::step], offsets_y[::step], offs_angles[yys, xxs][::step],
               width=0.0005, angles='xy', headwidth=6, cmap='jet', scale=10)

    # fig.canvas.draw()

    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    # return data

