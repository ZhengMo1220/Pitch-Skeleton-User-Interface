import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

from Widget.store_ui import store_ui_widget
import os
import cv2
import shutil
from utils.vis_pose import draw_points_and_skeleton, joints_dict
import pandas as pd

class Store_Widget(QtWidgets.QWidget):
    def __init__(self, video_name, video_images, person_df):
        super().__init__()
        self.ui = store_ui_widget()
        self.ui.setupUi(self)
        
        # 连接按钮的信号和槽
        self.ui.store_btn.clicked.connect(self.store)
        self.ui.cancel_btn.clicked.connect(self.cancel)
        self.person_df = person_df
        self.video_name = video_name
        self.video_images = video_images
        self.select_id = []
        
        self.saving_video = False
        self.add_id_checkbox()

    def store(self):
        self.save_datas()
        print("Store data success")
        self.close()

    def add_id_checkbox(self):
        person_ids = sorted(self.person_df['person_id'].unique())
        for person_id in person_ids:
            checkbox = QtWidgets.QCheckBox(f"{person_id}")
            checkbox.clicked.connect(lambda state, chk=checkbox: self.add_id_to_select(chk))
            self.ui.dispaly_id_layout.addWidget(checkbox)

    def add_id_to_select(self, checkbox):
        if checkbox.isChecked():
            # print(f"Checkbox with label '{checkbox.text()}' is checked.")
            self.select_id.append(int(checkbox.text()))
        else:
            self.select_id.remove(int(checkbox.text()))

    def reset_person_df(self):
        def reset_keypoints(keypoints):
            modified_keypoints = keypoints.copy()
            for kpt_idx, kpt in enumerate(keypoints):
                kptx, kpty = kpt[0], kpt[1]
                if kptx == 0.0 and kpty == 0.0:
                    modified_keypoints[kpt_idx][2] = 0
            return modified_keypoints

        self.person_df['keypoints'] = self.person_df['keypoints'].apply(reset_keypoints)

    def filter_person_df(self):
        filter_df = self.person_df[self.person_df['person_id'].isin(self.select_id)]
        return filter_df
    
    def save_datas(self):
        if self.saving_video:
            QMessageBox.warning(self, "儲存影片失敗", "請不要多次按下儲存影片按鈕!")
            return
        self.saving_video = True
        output_folder = f"../../Db/record/{self.video_name}"
        # if os.path.exists(output_folder):
        #     shutil.rmtree(output_folder)    
        os.makedirs(output_folder, exist_ok=True)
        
       # 将 DataFrame 保存为 JSON 文件
        json_path = os.path.join(output_folder, f"{self.video_name}.json")
        self.reset_person_df()

        if not self.select_id:
            save_person_df = self.person_df
        else:
            save_person_df = self.filter_person_df()

        save_person_df.to_json(json_path, orient='records')
        video_size = (1920, 1080)
        fps = 30.0
        save_location = "../../Db/record/" + self.video_name + "/" + self.video_name + "_Sk.mp4"

        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

        if not video_writer.isOpened():
            print("error while opening video writer!")
            return
        
        for frame_num, frame in enumerate(self.video_images):
            # if frame_num != 0:
            if frame is None or frame.shape[1] != video_size[0] or frame.shape[0] != video_size[1]:
                QMessageBox.critical(self, "錯誤", f"第 {frame_num} 幀圖像尺寸不匹配或無效！")
                # self.saving_video = False
                # return
            else:
                curr_person_df = self.obtain_frame_data(frame_num,save_person_df)
                image = self.draw_image(frame, curr_person_df)
                video_writer.write(image)

        video_writer.release()

        self.saving_video = False
        QMessageBox.information(self, "儲存影片", "影片儲存完成!")

            
    def draw_image(self,frame,person_df):    
        image=frame.copy()
        if not person_df.empty:
            image = draw_points_and_skeleton(image, person_df, joints_dict()['haple']['skeleton_links'], 
                                            points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                            points_palette_samples=10, confidence_threshold=0.3)
        return image
    
    def obtain_frame_data(self,frame_num,person_df):
        curr_person_df = pd.DataFrame()
        try :
            curr_person_df = person_df.loc[(person_df['frame_number'] == frame_num)]
        except ValueError:
            print("valueError")
        return curr_person_df

    def cancel(self):
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = Store_Widget()
    widget.show()
    sys.exit(app.exec_())
