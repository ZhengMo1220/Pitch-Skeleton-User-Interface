import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
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

def draw_analyze_infromation(image, analyze_information, jump_frame,
                              show_jump_speed, length_ratio,frame_ratio):
    # analyze_information = {'r_foot_kpt': [[x], [y], [peaks]],
    #                         'l_foot_kpt': [[x], [y], [peaks]],
    #                         'stride_time': [],
    #                         'stride_length': [],
    #                         'stride_pos': [],
    #                         'stride_speed': []}
    l_foot_kpt = analyze_information['l_foot_kpt']
    r_foot_kpt = analyze_information['r_foot_kpt']
    butt_kpt = analyze_information['butt_kpt']
    stride_time = analyze_information['stride_time']
    stride_pos = analyze_information['stride_pos']
    stride_length = analyze_information['stride_length']
    stride_speed = analyze_information['stride_speed']
    run_side = analyze_information['run_side']
    right_color = (240, 176, 0)
    left_color = (0,0,255)
    floor_point = [0,0]

    butt_kpt_is_exist = len(butt_kpt[0])>0 and len(butt_kpt[1]) >0

    if len(stride_pos)>0:
        floor_point = stride_pos[0]

    # draw stride point
    for time in stride_time:
        if time in l_foot_kpt[2]:
            x = l_foot_kpt[0][time]
            y = l_foot_kpt[1][time]
            draw_inverted_triangle(image, (int(x), int(y)), 20, left_color)
        elif time in r_foot_kpt[2]:
            x = r_foot_kpt[0][time]
            y = r_foot_kpt[1][time]
            draw_inverted_triangle(image, (int(x), int(y)), 20, right_color)

    image = draw_stride_length(image, stride_pos, stride_length)

    image = draw_stride_speed(image,stride_pos , stride_speed)

    image = draw_butt_point(image, butt_kpt, floor_point,length_ratio)

    image = draw_foot_point(image, l_foot_kpt, left_color)
    image = draw_foot_point(image, r_foot_kpt, right_color)

    # print(show_jump_speed)
    
    if show_jump_speed and butt_kpt_is_exist:
        jump_speed = [analyze_information['jump_horizontal_speed'] ,analyze_information['jump_vertical_speed']]
        image = draw_jump_speed(image, butt_kpt ,jump_speed, jump_frame,frame_ratio)

    return image

def draw_stride_length(img, stride_pos, stride_length):
    image = img.copy()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for i in range(0, len(stride_length)):
        x = int((stride_pos[i][0] + stride_pos[i+1][0]) / 2) - 120
        y = int((stride_pos[i][1] + stride_pos[i+1][1]) / 2) + 50
        text = "長度: {:.2f} 公尺".format(np.round(stride_length[i], 2))
        
        # 在 PIL 圖像上繪製文本
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                draw.text((x + dx, y + dy), text, font=fontStyle, fill=(0, 255, 0))
        
        # 將 PIL 圖像轉換回 OpenCV 圖像
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return image

def draw_stride_speed(img, stride_pos, stride_speed):
    image = img.copy()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 創建一個繪圖對象
    draw = ImageDraw.Draw(pil_image)

    for i in range(0, len(stride_speed)):
        x = int((stride_pos[i][0] + stride_pos[i+1][0]) / 2) - 120
        y = int((stride_pos[i][1] + stride_pos[i+1][1]) / 2) + 80
        text = "速度: {:.2f} 公尺/秒".format(np.round(stride_speed[i], 2))
        
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                draw.text((x + dx, y + dy) , text, font=fontStyle, fill=(0, 255, 0))
    
    # 將 PIL 圖像轉換回 OpenCV 圖像
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

def draw_butt_point(img,butt_kpt,floor_point,length_ratio):
    image = img.copy()
    i = 0
    color = (255, 0, 255)
    text_color = (255, 0, 255)
    for x, y in zip(butt_kpt[0], butt_kpt[1]):
        image = cv2.circle(image, (int(x), int(y)), 2, color, -1)
        pos_f = np.array([x, y])
        pos_s = np.array([x, floor_point[1]])
        if i % 20 == 0 and i > 0 and floor_point != [0, 0]:
            length = np.linalg.norm(pos_f - pos_s) * length_ratio
            text = "{:.2f} 公尺".format(np.round(length, 2))

            # 将 OpenCV 图像转换为 PIL 图像
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            # 确定文本位置
            text_x = int(pos_f[0])
            text_y = int((pos_f[1] + floor_point[1]) / 3)

            # 绘制加粗文本
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    draw.text((text_x + dx, text_y + dy), text, font=fontStyle, fill=text_color)

            # 将 PIL 图像转换回 OpenCV 图像
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        i += 1

    return image

def draw_foot_point(img,foot_kpt,color):
    image = img.copy()
    # i = 0
    for x,y in zip(foot_kpt[0], foot_kpt[1]):
        image = cv2.circle(image, (int(x), int(y)), 2, color, -1)
    return image

def draw_butt_width(image, person_kpt, select_frame, length_ratio, run_side):
    person_kpt = person_kpt.to_numpy()
    butt_pos = np.array([person_kpt[select_frame][19][0], person_kpt[select_frame][19][1]])
    if run_side:
        if person_kpt[select_frame][24][0] > person_kpt[select_frame][25][0]:
            ankle_pos = np.array([person_kpt[select_frame][24][0], person_kpt[select_frame][24][1]])
        else:
            ankle_pos = np.array([person_kpt[select_frame][25][0], person_kpt[select_frame][25][1]])
    else:
        if person_kpt[select_frame][24][0] > person_kpt[select_frame][25][0]:
            ankle_pos = np.array([person_kpt[select_frame][25][0], person_kpt[select_frame][25][1]])
        else:
            ankle_pos = np.array([person_kpt[select_frame][24][0], person_kpt[select_frame][24][1]])

    butt_width = np.abs(butt_pos[0] - ankle_pos[0]) * length_ratio * 100
    image = cv2.circle(image, (int(butt_pos[0]), int(butt_pos[1])), 5, (255, 0, 0), -1)
    draw_inverted_triangle(image, (int(butt_pos[0]), int(ankle_pos[1])), 20, (240, 176, 0))
    draw_inverted_triangle(image, (int(ankle_pos[0]), int(ankle_pos[1])), 20, (240, 176, 0))
    text = "{:.1f}公尺".format(np.round(butt_width, 1))

    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 确定文本位置
    text_x = int(butt_pos[0])
    text_y = int(ankle_pos[1]) - 30

    # 绘制加粗文本
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            draw.text((text_x + dx, text_y + dy), text, font=fontStyle, fill=(0, 0, 255))

    # 将 PIL 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
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

def draw_jump_speed(img,butt_kpt,jump_speed,jump_frame,frame_ratio):
    # jump_speed = [jump_horizontal_speed , jump_vertical_speed]
    image = img.copy()
    color = (140, 199, 0)
    text_color = (0, 255, 127)
    butt_kpt = [butt_kpt[0][jump_frame[0]:jump_frame[1]], butt_kpt[1][jump_frame[0]:jump_frame[1]]]
    for x,y in zip(butt_kpt[0], butt_kpt[1]):
        image = cv2.circle(image, (int(x), int(y)), 2, color, -1)

    draw_inverted_triangle(image, (int(butt_kpt[0][0]), int(butt_kpt[1][0])), 20, color)
    h_mean = np.mean(jump_speed[0])
    v_mean = np.mean(jump_speed[1])
    t = (jump_frame[1] - jump_frame[0]) * frame_ratio
    draw_inverted_triangle(image, (int(butt_kpt[0][-1]), int(butt_kpt[1][-1])), 20, color)
    h_text = "水平速度: {:.2f} 公尺/秒".format(np.round(h_mean, 2))
    v_text = "垂直速度: {:.2f} 公尺/秒".format(np.round(v_mean, 2))
    t_text = "滯空時間: {:.2f}      秒".format(np.round(v_mean, 2))
    x = int(butt_kpt[0][0] / 2 + butt_kpt[0][1] / 2)
    y = int(butt_kpt[1][0]) 

    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 绘制加粗文本
    for dx in range(-1, 1):
        for dy in range(-1, 1):
            draw.text((x + dx, y + 30 + dy), h_text, font=fontStyle, fill=text_color)
            draw.text((x + dx, y + 60 + dy), v_text, font=fontStyle, fill=text_color)
            draw.text((x + dx, y + 90 + dy), t_text, font=fontStyle, fill=text_color)

    # 将 PIL 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return image
