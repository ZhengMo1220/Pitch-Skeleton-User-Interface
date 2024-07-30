import cv2

# 初始化 VideoCapture 對象，參數 0 表示默認攝像頭
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝像頭")
    exit()

# 連續讀取影像
while True:
    # 讀取一幀影像
    ret, frame = cap.read()

    # 如果讀取成功
    if ret:
        # 顯示影像
        cv2.imshow('Camera', frame)

        # 按下 'q' 鍵退出循環
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("無法讀取影像")
        break

# 釋放攝像頭資源
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
