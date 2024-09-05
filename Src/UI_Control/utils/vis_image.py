import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from scipy.signal import savgol_filter
import sys
try:
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow').colors) * 255
    ).astype(np.uint8)[:, ::-1].tolist()
except AttributeError:  # if palette has not pre-defined colors
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow')(np.linspace(0, 1, 10))) * 255
    ).astype(np.uint8)[:, -2::-1].tolist()

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(__file__)

font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
print("Font path:", font_path)  # 打印以确认路径
fontStyle = ImageFont.truetype(font_path, 20)

def draw_cross(image, x, y, length=5, color=(0, 0, 255), thickness=2):
    cv2.line(image, (x, y - length), (x, y + length), color, thickness)
    cv2.line(image, (x - length, y), (x + length, y), color, thickness)

def draw_grid(image:np.ndarray):
    #return image:np.ndarray
    
    height, width = image.shape[:2]


    draw_cross(image,int(width/2),int(height/2),length=20,color=(0,0,255),thickness = 3)


    # 計算垂直線的位置
    vertical_interval = width // 5
    vertical_lines = [vertical_interval * i for i in range(1, 5)]

    # 計算水平線的位置
    horizontal_interval = height // 5
    horizontal_lines = [horizontal_interval * i for i in range(1, 5)]

    # 畫垂直線
    for x in vertical_lines:
        cv2.line(image, (x, 0), (x, height), (0, 255, 0), 2)

    # 畫水平線
    for y in horizontal_lines:
        cv2.line(image, (0, y), (width, y), (0, 255, 0), 2)

    return image


def draw_bbox(person_data, image):
    person_ids = person_data['person_id']
    person_bbox = person_data['bbox']
    for id, bbox in zip(person_ids, person_bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = tuple(colors[id % len(colors)])
        color = (0,255,0)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
    return image

def draw_inverted_triangle(image, center, size, color, border_color=(0, 0, 0), border_thickness=2):
    """
    在图像上绘制一个倒三角形，并添加黑色边框。

    参数:
    - image: 输入图像。
    - center: 三角形的底部顶点（x, y）。
    - size: 三角形的边长。
    - color: 三角形的颜色（B, G, R）。
    - border_color: 边框的颜色（默认黑色）。
    - border_thickness: 边框的粗细（默认2）。
    """
    height = int(size * (np.sqrt(3) / 2))  # 计算三角形的高度
    vertices = np.array([
        [center[0], center[1]],  # 底部顶点
        [center[0] - size // 2, center[1] - height],  # 左上顶点
        [center[0] + size // 2, center[1] - height]   # 右上顶点
    ], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    
    # 先画边框
    cv2.polylines(image, [vertices], isClosed=True, color=border_color, thickness=border_thickness)
    # 填充三角形
    cv2.fillPoly(image, [vertices], color)

def draw_traj(kpt_buffer, img):
    if len(kpt_buffer) == 0 or len(kpt_buffer) == 1:
        return img
    image = img.copy()
    
    for i in range(1, len(kpt_buffer)):
        f_kptx, f_kpty = map(int, kpt_buffer[i-1])
        s_kptx, s_kpty = map(int, kpt_buffer[i])
        cv2.line(image, (f_kptx, f_kpty), (s_kptx, s_kpty), (0, 255, 0), 5)

    return image

def draw_video_traj(img, person_df, person_id, kpt_id, frame_num, window_length=17, polyorder=2):
    """
    在圖片上繪製特定人物的關鍵點軌跡，並使用Savgol濾波器平滑軌跡。

    參數：
    - img: 原始圖片（NumPy陣列）
    - person_df: 包含人物關鍵點信息的Pandas DataFrame
    - person_id: 目標人物的ID
    - kpt_id: 目標關鍵點的ID
    - frame_num: 總幀數
    - window_length: Savgol濾波器的窗口長度（默認17，需為奇數）
    - polyorder: Savgol濾波器的多項式階數（默認2）
    
    返回：
    - image: 帶有繪製軌跡的圖片
    """
    
    # 如果DataFrame為空，直接返回原圖
    if person_df.empty:
        return img.copy()
    
    # 複製原始圖片以避免修改原圖
    image = img.copy()
    
    # 過濾DataFrame，僅保留目標人物和指定幀數範圍內的數據
    filtered_df = person_df[
        (person_df['person_id'] == person_id) & 
        (person_df['frame_number'] < frame_num)
    ]
    
    # 如果過濾後的DataFrame為空，返回原圖
    if filtered_df.empty:
        return image
    
    # 按幀數排序，確保軌跡的連貫性
    filtered_df = filtered_df.sort_values(by='frame_number')
    
    # 提取指定關鍵點的(x, y)座標
    # 假設 'keypoints' 列包含一個列表，每個元素是關鍵點的四元組 (x, y, visibility, ... )
    kpt_buffer = [
        (kpt[0], kpt[1]) 
        for kpts in filtered_df['keypoints'] 
        if kpts and kpt_id < len(kpts) and kpts[kpt_id] is not None
        for kpt in [kpts[kpt_id]]
    ]
    
    # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
    if len(kpt_buffer) >= window_length:
        # 確保窗口長度為奇數且不超過緩衝區長度
        if window_length > len(kpt_buffer):
            window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
        # 確保多項式階數小於窗口長度
        current_polyorder = min(polyorder, window_length - 1)
        
        # 分別提取x和y座標
        x = np.array([point[0] for point in kpt_buffer])
        y = np.array([point[1] for point in kpt_buffer])
        
        # 應用Savgol濾波器
        x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)
        
        # 將平滑後的座標重新打包
        smoothed_points = list(zip(x_smooth, y_smooth))
    else:
        # 緩衝區長度不足，直接使用原始座標
        smoothed_points = kpt_buffer
    
    # 如果平滑後的點少於2個，無需繪製軌跡
    if len(smoothed_points) < 2:
        return image
    
    # 將座標轉換為整數並構造適合cv2.polylines的格式
    points = np.array([(int(x), int(y)) for x, y in smoothed_points], dtype=np.int32).reshape(-1, 1, 2)
    
    # 使用cv2.polylines繪製連續的軌跡線
    cv2.polylines(image, [points], isClosed=False, color=(0, 255, 0), thickness=5)
    
    return image


def draw_angle_info(img: np.ndarray, angle_info: pd.DataFrame, frame_num:int, pos:tuple):
    image = img.copy()

    if angle_info is None:
        return image
    data = angle_info.loc[angle_info['frame_number'] == (frame_num-1)]
    data = data['angle'].iloc[0]

    for _, info in data.items(): 
        angle = int(info[0])
        pt1, pt2, pt3 = [tuple(map(int, point)) for point in info[1]]
        if pos is not None:
            cv2.line(image, pt2, (pos[0] +20 , pos[1] + 180), (0, 0, 0), 1)
            image = cv2.putText(image, str(angle), (pos[0] - 20, pos[1] + 200), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
    
    return image


def draw_region(img:np.ndarray):
    image = img.copy()
    cv2.rectangle(image, (100, 250), (450, 600), (0, 255, 0), -1)
    return image