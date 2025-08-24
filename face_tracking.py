import cv2
import numpy as np
import time

def try_video_capture(max_retries=3):
    """尝试打开不同的摄像头索引"""
    for camera_index in [0, 1, 2]:  # 尝试不同的摄像头索引
        for _ in range(max_retries):
            print(f"尝试打开摄像头 {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            if cap is None or not cap.isOpened():
                print(f"摄像头 {camera_index} 打开失败，等待重试...")
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"成功打开摄像头 {camera_index}")
                return cap
            
            cap.release()
            time.sleep(1)
    
    return None

def main():
    # 尝试打开摄像头
    cap = try_video_capture()
    
    if cap is None:
        print("无法打开任何摄像头，请检查摄像头权限和连接状态")
        return
    
    # 加载人脸检测的级联分类器
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("无法加载人脸检测模型")
    except Exception as e:
        print(f"加载人脸检测模型失败: {e}")
        cap.release()
        return
    
    print("摄像头已就绪，按'q'键退出程序")
    
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret or frame is None:
            print("无法获取摄像头画面")
            break
            
        try:
            # 水平翻转画面
            frame = cv2.flip(frame, 1)  # 1表示水平翻转
            
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # 在检测到的人脸周围画矩形
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # 显示结果
            cv2.imshow('Face Tracking', frame)
            
        except Exception as e:
            print(f"处理帧时发生错误: {e}")
            break
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    print("正在关闭程序...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
