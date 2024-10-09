import numpy as np
import pandas as pd
from skeleton.detect_skeleton import PoseEstimater

class PoseAnalyzer:
    def __init__(self, pose_estimater:PoseEstimater):
        self.pose_estimater = pose_estimater
        self.angle_dict = self.pose_estimater.joints['haple']['angle_dict']
        self.analyze_info = []
        self.analyze_df = pd.DataFrame()
        self.processed_frames = set()

    def addAnalyzeInfo(self, frame_num: int):
        """Analyze information for each frame up to the current frame."""
        if self.pose_estimater.person_id is None:
            return pd.DataFrame()
        
        person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True,is_kpt=True)
        
        if person_kpt is None:
            return
        info = {
            'frame_number': frame_num,
            'angle': self._update_analyze_information(person_kpt)
        }

        if frame_num not in self.processed_frames:
            self.processed_frames.add(frame_num)
            self.analyze_info.append(info)
            self.analyze_df = pd.DataFrame(self.analyze_info)
            self.analyze_df = self.analyze_df.sort_values(by='frame_number').reset_index(drop=True)

    def _calculate_angle(self, A, B, C):
        """Calculate the angle between three points A, B, and C."""
        BA = np.array(A) - np.array(B)
        BC = np.array(C) - np.array(B)
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        cos_angle = dot_product / (magnitude_BA * magnitude_BC)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def _update_analyze_information(self, person_kpt):
        """Update and return analyze information for the given keypoints."""
        info = {}
        for angle_name, kpt_list in self.angle_dict.items():
            A = person_kpt[kpt_list[0]][:2]
            B = person_kpt[kpt_list[1]][:2]
            C = person_kpt[kpt_list[2]][:2]
            info[angle_name] = [self._calculate_angle(A, B, C), [np.array(A), np.array(B), np.array(C)]]
        return info

    def get_frame_angle_data(self, frame_num: int = None, angle_name: str = None):
        if self.analyze_df.empty:
            return pd.DataFrame(), []
        condition = pd.Series([True] * len(self.analyze_df))

        # 根據 frame_num 過濾數據
        if frame_num is not None:
            condition &= (self.analyze_df['frame_number'] == frame_num)

        data = self.analyze_df.loc[condition]
        
        # 如果數據為空則返回
        if data.empty:
            return None, []

        if angle_name is not None:
            # 如果指定了 angle_name 且有特定幀數，返回該幀的角度數據
            if frame_num is not None:
                angle_value = data['angle'].iloc[0][angle_name]
                return data, angle_value
            else:
                # 如果未指定 frame_num，則返回該 angle_name 的所有幀數及對應角度
                frame_numbers = self.analyze_df['frame_number'].unique()
                angles = [row['angle'][angle_name][0] for _, row in self.analyze_df.iterrows() if angle_name in row['angle']]
                return frame_numbers, angles
        
        return data, []

    def reset(self):
        self.analyze_info = []
        self.analyze_df = pd.DataFrame()
        self.processed_frames = set()

class JointAreaChecker:
    def __init__(self, image_size:tuple):
        self.image_width = image_size[0]
        self.image_height = image_size[1]

        # 設置1/5區域的邊界
        self.region_width = image_size[0]
        self.region_height = image_size[1] // 5

        # 區域左上角的坐標
        self.region_top_left = (0, 0)
        self.region_bottom_right = (self.region_width, self.region_height)

    def is_joint_in_area(self, joint_position:tuple)->bool:
        """檢查關節點是否在定義的區域內"""
        if joint_position is None:
            return False
        x, y = joint_position
        
        in_area = (self.region_top_left[0] <= x <= self.region_bottom_right[0] and
                   self.region_top_left[1] <= y <= self.region_bottom_right[1])
        return in_area