import numpy as np
from scipy.signal import find_peaks, savgol_filter

def obtain_analyze_information(person_kpt, angle_dict):
    person_kpt = person_kpt.to_numpy()[0]
    analyze_information = initialize_analyze_information()
    analyze_information = update_analyze_information(analyze_information, person_kpt, angle_dict)

    return analyze_information

def initialize_analyze_information():
    return {
        'l_elbow_angle': [],
        'r_elbow_angle': [],
        'l_shoulder_angle': [],
        'r_shoulder_angle': [],
        'l_knee_angle': [],
        'r_knee_angle': []
        }


def calculate_angle(A, B, C):
    # 將點座標轉換為向量
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    # 計算向量的點積
    dot_product = np.dot(BA, BC)
    
    # 計算向量的模長
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # 計算角度的餘弦值
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    
    # 使用 arccos 計算角度（以弧度為單位），然後轉換為度數
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def update_analyze_information(analyze_information, person_kpt, angle_dict):

    for angle_name, kpt_list in angle_dict.items():
        A = person_kpt[kpt_list[0]][:2]
        B = person_kpt[kpt_list[1]][:2]
        C = person_kpt[kpt_list[2]][:2]
        analyze_information[angle_name] = [calculate_angle(A, B, C), [np.array(A), np.array(B), np.array(C)]]

    return analyze_information
