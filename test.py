import cv2
import numpy as np

# 建立一個空白影像
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 設定兩個點
pt1 = (200, 50)
pt2 = (400, 300)

# 計算弧線的中心點
center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

# 計算半徑
radius = int(np.linalg.norm(np.array(pt1) - np.array(center)))

# 設定起始角度和結束角度
start_angle = 0
end_angle = 180
print(center)
print(radius)
# 繪製弧線
cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, (255, 255, 255), 2)

# 顯示影像
cv2.imshow('Arc', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
