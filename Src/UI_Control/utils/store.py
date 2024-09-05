import os
import cv2
from utils.vis_pose import draw_points_and_skeleton, joints_dict
import pandas as pd

def obtain_frame_data(frame_num, person_df):
    """取得特定幀數的資料。"""
    try:
        return person_df.loc[person_df['frame_number'] == frame_num]
    except ValueError:
        print("ValueError occurred while obtaining frame data.")
        return pd.DataFrame()
    
def obtain_data(person_df, frame_num=None, person_id=None):
    condition = pd.Series([True] * len(person_df))  # 初始條件設為全為 True
    if frame_num is not None:
        condition &= (person_df['frame_number'] == frame_num)
    
    if person_id is not None:
        condition &= (person_df['person_id'] == person_id)

    data = person_df.loc[condition]
        
    return data

def reset_keypoints(keypoints):
    """重設關鍵點的標籤。"""
    return [[x, y, 0] if x == 0.0 and y == 0.0 else [x, y, conf] for x, y, conf, _ in keypoints]

def reset_person_df(person_df):
    """重設所有人的關鍵點資料。"""
    person_df['keypoints'] = person_df['keypoints'].apply(reset_keypoints)
    return person_df

def filter_person_df(person_df, select_id):
    """過濾出特定人的資料。"""
    return person_df[person_df['person_id'] == select_id]

def draw_image(frame, person_df):
    """在影像上繪製關鍵點和骨架。"""
    image = frame.copy()
    if not person_df.empty:
        image = draw_points_and_skeleton(image, person_df, joints_dict()['haple']['skeleton_links'],
                                         points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                         points_palette_samples=10, confidence_threshold=0.3)
    return image

def save_video(video_name, video_images, person_df, select_id=None):
    """儲存影片及其對應的 JSON 檔案。"""

    output_folder = os.path.join("../../Db/Record", video_name)
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(output_folder, f"{video_name}.json")
    save_person_df = person_df.copy()
    # if select_id is not None:
    #     save_person_df = filter_person_df(save_person_df, select_id)
    
    save_person_df.to_json(json_path, orient='records')

    video_size = (video_images[0].shape[1], video_images[0].shape[0])
    fps = 30.0
    save_location = os.path.join(output_folder, f"{video_name}_Sk26.mp4")

    video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

    if not video_writer.isOpened():
        print("Error while opening video writer!")
        return

    for frame_num, frame in enumerate(video_images):
        if frame is not None and frame.shape[:2] == (video_size[1], video_size[0]):
            curr_person_df = obtain_data(person_df, frame_num=frame_num, person_id=select_id)
            image = draw_image(frame, curr_person_df)
            video_writer.write(image)

    video_writer.release()
    print("Store video success")
