import cv2
import matplotlib.pyplot as plt
import numpy as np


def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "鼻子",
                1: "左眼",
                2: "右眼",
                3: "左耳",
                4: "右耳",
                5: "左肩",
                6: "右肩",
                7: "左手肘",
                8: "右手肘",
                9: "左手腕",
                10: "右手腕",
                11: "左髋",
                12: "右髋",
                13: "左膝",
                14: "右膝",
                15: "左腳踝",
                16: "右腳踝"
            },
            "skeleton_links": [
                [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                #軀幹
                [5, 7], [7, 9],                 #左手
                [6, 8], [8, 10],                #右手
                [11, 13], [13, 15],   #左腿
                [12, 14], [14, 16],   #右腿
            ],
            "left_points_indices": [[5, 7], [7, 9], [11, 13], [13, 15]],  # Indices of left hand, leg, and foot keypoints
            "right_points_indices": [[6, 8], [8, 10], [12, 14], [14, 16]]  # Indices of right hand, leg, and foot keypoints
        },
        "haple":{
            "keypoints": {
                0: "鼻子",
                1: "左眼",
                2: "右眼",
                3: "左耳",
                4: "右耳",
                5: "左肩",
                6: "右肩",
                7: "左肘",
                8: "右肘",
                9: "左腕",
                10: "右腕",
                11: "左髖",
                12: "右髖",
                13: "左膝",
                14: "右膝",
                15: "左踝",
                16: "右踝",
                17: "頭部",
                18: "頸部",
                19: "臀部",
                20: "左大腳趾",
                21: "右大腳趾",
                22: "左小腳趾",
                23: "右小腳趾",
                24: "左腳跟",
                25: "右腳跟"
            },
            "skeleton_links":[
                [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                [5, 18], [6, 18], [17, 18],[18, 19],#軀幹
                [5, 7], [7, 9],                 #左手
                [6, 8], [8, 10],                #右手
                [19, 11], [11, 13], [13, 15],   #左腿
                [19, 12], [12, 14], [14, 16],   #右腿
                [20, 24], [22, 24], [15, 24],   #左腳
                [21, 25], [23, 25], [16, 25]    #右腳
            ],
            
            "left_points_indices": [[5, 18], [5, 7], [7, 9],[19, 11], [11, 13], [13, 15], [20, 24], [22, 24], [15, 24]],  # Indices of left hand, leg, and foot keypoints
            "right_points_indices": [[6, 18], [6, 8], [8, 10], [19, 12], [12, 14], [14, 16], [21, 25], [23, 25], [16, 25]]  # Indices of right hand, leg, and foot keypoints
        },
    }
    return joints

def draw_points(image, points, person_idx, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
    for i, pt in enumerate(points):
      
        unlabel = False if pt[0] != 0 and pt[1] != 0 else True
        if pt[2] > confidence_threshold and not unlabel:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[person_idx % len(colors)]), -1)

    return image

def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()
    right_skeleton = joints_dict()['haple']['right_points_indices']
    left_skeleton = joints_dict()['haple']['left_points_indices']
    
    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        pt1_unlabel = False if pt1[0] != 0 and pt1[1] != 0 else True
        pt2_unlabel = False if pt2[0] != 0 and pt2[1] != 0 else True
        skeleton_color = tuple(colors[person_index % len(colors)])
        skeleton_color = (0,255,0)
        if joint in right_skeleton:
            skeleton_color = (240, 176, 0)
        elif joint in left_skeleton:
            skeleton_color = (0, 0, 255)
        if pt1[2] > confidence_threshold and not pt1_unlabel and pt2[2] > confidence_threshold and not pt2_unlabel:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                skeleton_color , 4
            )
    return image

def draw_points_and_skeleton(image, person_df, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, confidence_threshold=0.5):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    person_data = df_to_points(person_df)
    for person_id, points in person_data.items(): 
        image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                            palette_samples=skeleton_palette_samples, person_index=person_id,
                            confidence_threshold=confidence_threshold)
        image = draw_points(image, points,person_idx=person_id, color_palette=points_color_palette, palette_samples=points_palette_samples,
                            confidence_threshold=confidence_threshold)
    return image

def df_to_points(person_df):
    person_data = {}
    person_ids = person_df['person_id']
    person_kpts = person_df['keypoints']
    for id, kpts in zip(person_ids, person_kpts):
        person_data[id] = np.array(swap_values(kpts))
    return person_data

def swap_values(kpts):
    return [[item[1], item[0], item[2]] for item in kpts]

def draw_tracking_skeleton(image, person_kpt, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, confidence_threshold=0.5):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(image, person_kpt, skeleton, color_palette=skeleton_color_palette,
                        palette_samples=skeleton_palette_samples, person_index=0,
                        confidence_threshold=confidence_threshold)
    image = draw_points(image, person_kpt,person_idx=0, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
    return image
