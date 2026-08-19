class PitchingAnalyzer:
    def __init__(self):
        self.state = "IDLE"
        self.offset = 30  # 像素門檻，視攝影機距離調整
        self.is_active = False
        self.frame_count = 0  # 帧计数器
        self.init_frames = 12  # 初始化缓冲帧数（约1秒）

    def process(self, keypoints):
        """
        輸入：PoseEstimater 輸出的 keypoints (26, 4)
        輸出：action_signal ("START", "STOP", None)
        """
        if keypoints is None:
            return None

        # 初始化缓冲期：跳过前12帧
        if self.frame_count < self.init_frames:
            self.frame_count += 1
            return None

        # 提取 y 座標 (注意：影像座標系 y 向下為正)
        r_wrist_y = keypoints[10][1]
        r_hip_y = keypoints[12][1] # 右肩
        l_knee_y = keypoints[13][1] # 左腕
        neck_y = keypoints[18][1] # 鼻子

        signal = None

        # 狀態機切換邏輯
        if self.state == "IDLE":
            # 偵測抬腿 (膝蓋向上移動，y 值變小)
            # print(f"[Debug] r_wrist_y: {r_wrist_y}, r_hip_y: {r_hip_y}")
            if l_knee_y < (r_hip_y + self.offset):
                self.state = "KNEEUP"
                self.is_active = True
                signal = "START"
                print("[Action] 偵測到抬腿，開始錄影...")

        elif self.state == "KNEEUP":
            # 偵測到手部舉起，進入投球階段
            if r_wrist_y < neck_y:
                self.state = "DELIVERY"

        elif self.state == "DELIVERY":
            # 偵測到手部落下，動作結束
            if r_wrist_y > (r_hip_y - self.offset):
                self.state = "IDLE"
                self.is_active = False
                self.frame_count = 0  # 重置帧计数器
                signal = "STOP"
                print("[Action] 投球結束，準備存檔回傳。")

        return signal

    def reset(self):
        self.state = "IDLE"
        self.is_active = False
        self.frame_count = 0  # 重置帧计数器