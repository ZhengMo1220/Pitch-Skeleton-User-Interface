import numpy as np
from scipy.signal import find_peaks, savgol_filter

def obtain_analyze_information(person_kpt, curr_frame_num, process_frame_num,
                                length_ratio, frame_ratio, jump_frame, line_pos, end_frame):
    person_kpt = person_kpt.to_numpy()
    analyze_information = initialize_analyze_information()
    if curr_frame_num > end_frame:
        end_analyze_frame_num = end_frame - process_frame_num + 1
        # print(end_analyze_frame_num)
    else:
        end_analyze_frame_num = curr_frame_num - process_frame_num + 1
        # print(end_analyze_frame_num)
    
    finish_fly_frame = 0
    
    if len(person_kpt) == end_analyze_frame_num or len(person_kpt) > end_analyze_frame_num:
        for frame_num in range(0, end_analyze_frame_num):
            update_analyze_information(analyze_information, person_kpt, frame_num)

    if len(analyze_information['l_foot_kpt'][0]) > 0:
        run_side = calculate_run_side(analyze_information['butt_kpt'][0][0])
    else:
        run_side = False
    print("左進") if run_side else print("右進")
    analyze_information['l_foot_kpt'][2] = obtain_peaks(analyze_information['l_foot_kpt'], run_side, line_pos)
    analyze_information['r_foot_kpt'][2] = obtain_peaks(analyze_information['r_foot_kpt'], run_side, line_pos)
    stride_lengths, stride_speeds, stride_time, stride_pos = process_peaks(analyze_information, length_ratio, frame_ratio)
    analyze_information['stride_length'] = stride_lengths
    analyze_information['stride_time'] = stride_time
    analyze_information['stride_speed'] = stride_speeds
    analyze_information['stride_pos'] = stride_pos
    if jump_frame[:2] != [0,0] and curr_frame_num > jump_frame[0]:
        # print(jump_frame)
        jump_frame = [jump_frame[0] - process_frame_num , jump_frame[1] - process_frame_num]
        curr_frame = curr_frame_num - process_frame_num
        jump_speed = calculate_jump_speeds(analyze_information['butt_kpt'], curr_frame,jump_frame,length_ratio,frame_ratio)
        fly_speed, finish_fly_frame = calculate_fly_speeds(analyze_information['butt_kpt'], curr_frame,jump_frame,
                                                            length_ratio,frame_ratio,line_pos,run_side)
        # print(jump_speed)
        analyze_information['jump_horizontal_speed'] = jump_speed[0]
        analyze_information['jump_vertical_speed'] = jump_speed[1]
        analyze_information['fly_horizontal_speed'] = fly_speed[0]
        analyze_information['fly_vertical_speed'] = fly_speed[1]
        analyze_information['run_side'] = run_side

    return analyze_information, finish_fly_frame

def initialize_analyze_information():
    return {
        'r_foot_kpt': [[], [], []],
        'l_foot_kpt': [[], [], []],
        'butt_kpt':[[],[]],
        'stride_time': [],
        'stride_length': [],
        'stride_speed': [],
        'stride_pos': [],
        'jump_horizontal_speed': [],
        'jump_vertical_speed': [],
        'fly_horizontal_speed': 0, 
        'fly_vertical_speed': 0,
        'run_side': False
    }

def update_analyze_information(analyze_information, person_kpt, frame_num):
    analyze_information['l_foot_kpt'][0].append(person_kpt[frame_num][20][0])
    analyze_information['l_foot_kpt'][1].append(person_kpt[frame_num][20][1])
    analyze_information['r_foot_kpt'][0].append(person_kpt[frame_num][21][0])
    analyze_information['r_foot_kpt'][1].append(person_kpt[frame_num][21][1])
    analyze_information['butt_kpt'][0].append(person_kpt[frame_num][19][0])
    analyze_information['butt_kpt'][1].append(person_kpt[frame_num][19][1])

def obtain_peaks(data, run_side, line_pos):
    peaks = []
    # processed_data = process_data(data, run_side, line_pos)
    processed_data = data[1]
    smooth_data = processed_data
    # if len(processed_data)>11 :
        # smooth_data = savgol_filter(np.array(processed_data), window_length=11, polyorder=3)
    peaks, _ = find_peaks(smooth_data, width=8, prominence=5)
        # peaks, _ = find_peaks(processed_data, prominence=5)
    # print(peaks)
    return peaks

def process_data(data, run_side, line_pos):

    processed_data = []

    if run_side:
        end_line_pos = max(line_pos[0], line_pos[2]) + 20
        compare_func = lambda x: x < end_line_pos
    else:
        end_line_pos = min(line_pos[0], line_pos[2]) - 20
        compare_func = lambda x: x > end_line_pos

    for x, y in zip(data[0], data[1]):
        if compare_func(x):
            processed_data.append(y)

    return processed_data

def process_peaks(analyze_information,length_ratio,frame_ratio):
    stride_pos = []
    stride_time = []
    stride_lengths = []
    stride_speeds = []

    l_peaks = analyze_information['l_foot_kpt'][2]
    r_peaks = analyze_information['r_foot_kpt'][2]
    stride_time = sorted([*l_peaks,*r_peaks])
    is_stride_exist = len(stride_time) > 1

    if is_stride_exist:
        for time in stride_time:
            if time in l_peaks:
                stride_pos.append(get_position(analyze_information['l_foot_kpt'], time))
            elif time in r_peaks:
                stride_pos.append(get_position(analyze_information['r_foot_kpt'], time))
        
        stride_lengths = calculate_stride_lengths(stride_pos, length_ratio)
        stride_speeds = calculate_stride_speeds(stride_lengths, stride_time, frame_ratio)

    return stride_lengths, stride_speeds, stride_time, stride_pos

def get_position(foot_kpt, index):
    return (foot_kpt[0][index], foot_kpt[1][index])

def calculate_stride_lengths(stride_pos, length_ratio):
    stride_lengths = []
    for i in range(1, len(stride_pos)):
        stride_length = np.linalg.norm(np.array(stride_pos[i]) - np.array(stride_pos[i - 1])) * length_ratio
        stride_lengths.append(stride_length)
    return stride_lengths

def calculate_stride_speeds(stride_lengths, stride_time, frame_ratio):
    stride_speeds = []
    
    for i in range(1, len(stride_time)):
        time_diff = (stride_time[i] - stride_time[i - 1]) * frame_ratio
        
        speed = stride_lengths[i - 1] / time_diff 
    
        stride_speeds.append(speed)
    return stride_speeds

def calculate_jump_speeds(kpt, curr_frame,jump_frame,length_ratio,frame_ratio):
    vertical_speeds = []
    horizontal_speeds = []
    start = jump_frame[0]
    end = min(jump_frame[1], curr_frame) - 1
    kpt_is_exist=  len(kpt[0]) >0 and len(kpt[1]) > 0

    if kpt_is_exist:
        for i in range(start, end):
            time_diff = 1 * frame_ratio
            vertical_length = np.abs(kpt[1][i] - kpt[1][i+1]) * length_ratio
            horizontal_length = np.abs(kpt[0][i] - kpt[0][i+1]) * length_ratio
            vertical_speed = vertical_length / time_diff
            horizontal_speed = horizontal_length / time_diff
            vertical_speeds.append(vertical_speed)
            horizontal_speeds.append(horizontal_speed)

    return [horizontal_speeds , vertical_speeds]

def calculate_fly_speeds(kpt, curr_frame,jump_frame,length_ratio,frame_ratio, line_pos,run_side):
    vertical_speed = 0
    horizontal_speed = 0
    kpt_is_exist=  len(kpt[0]) >0 and len(kpt[1]) > 0
    if run_side:
        end_line_pos = max(line_pos[0], line_pos[2])
    else:
        end_line_pos = min(line_pos[0], line_pos[2])
    start = jump_frame[0]
    end = find_finish_frame(kpt[0], end_line_pos,run_side)

    if kpt_is_exist and end != -1:
        time_diff = (end - start) * frame_ratio
        vertical_length = np.abs(kpt[1][end] - kpt[1][start]) * length_ratio
        horizontal_length = np.abs(kpt[0][end] - kpt[0][start]) * length_ratio
        vertical_speed = vertical_length / time_diff
        horizontal_speed = horizontal_length / time_diff

    return [horizontal_speed , vertical_speed], end

def find_finish_frame(numbers, threshold,run_side):
    for index, value in enumerate(numbers):
        if run_side:
            if value > threshold:
                return index
        else:
            if value < threshold:
                return index
    return -1  # 如果没有找到大于threshold的值，返回-1

def calculate_run_side(foot_kpt_pos):
    # True: left in, False: right in
    if  foot_kpt_pos < 600:
        return True
    else:
        return False