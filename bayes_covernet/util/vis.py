'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
import cv2
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.utils import (
    convert_to_pixel_coords,
    get_crops,
    get_rotation_matrix,
)
from pyquaternion.quaternion import Quaternion

import numpy as np


def draw_global_pts_in_image(
    img, agent_translation, agent_rotation, agent_rasterizer, global_pts, color
):
    image_side_length = 2 * max(
        agent_rasterizer.meters_ahead,
        agent_rasterizer.meters_behind,
        agent_rasterizer.meters_left,
        agent_rasterizer.meters_right,
    )
    image_side_length_pixels = int(image_side_length / agent_rasterizer.resolution)
    agent_pixels = int(image_side_length_pixels / 2), int(image_side_length_pixels / 2)
    row_crop, col_crop = get_crops(
        agent_rasterizer.meters_ahead,
        agent_rasterizer.meters_behind,
        agent_rasterizer.meters_left,
        agent_rasterizer.meters_right,
        agent_rasterizer.resolution,
        int(image_side_length / agent_rasterizer.resolution),
    )

    agent_yaw = quaternion_yaw(Quaternion(agent_rotation))
    rotation_mat = get_rotation_matrix(
        (image_side_length_pixels, image_side_length_pixels), agent_yaw
    )

    for start_global, end_global in zip(global_pts[:-1], global_pts[1:]):
        start_pixels = convert_to_pixel_coords(
            start_global, agent_translation, agent_pixels, agent_rasterizer.resolution
        )
        end_pixels = convert_to_pixel_coords(
            end_global, agent_translation, agent_pixels, agent_rasterizer.resolution
        )

        start_pixels = (start_pixels[1], start_pixels[0])
        end_pixels = (end_pixels[1], end_pixels[0])

        start_pixels = rotation_mat.dot(
            np.hstack([np.asarray(start_pixels), 1]).T
        ).T.astype(int)
        end_pixels = rotation_mat.dot(
            np.hstack([np.asarray(end_pixels), 1]).T
        ).T.astype(int)

        start_pixels = (
            start_pixels[0] - col_crop.start,
            start_pixels[1] - row_crop.start,
        )
        end_pixels = (end_pixels[0] - col_crop.start, end_pixels[1] - row_crop.start)

        # =======================================================================
        # if start_pixels[0] >= row_crop.stop or start_pixels[1] >= col_crop.stop or end_pixels[0] >= row_crop.stop or end_pixels[1] >= col_crop.stop:
        #     continue
        # =======================================================================

        cv2.line(img, start_pixels, end_pixels, color, thickness=2)
