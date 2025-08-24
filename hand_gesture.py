import cv2
import numpy as np

# 手势识别器类，使用 OpenCV 实现
class HandGestureRecognizer:
    def __init__(self):
        # 初始化肤色检测的HSV范围
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def detect_hand(self, frame):
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建肤色掩码
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 形态学操作，去除噪点
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5,5), 100)
        
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有检测到轮廓，返回None
        if not contours:
            return None, None
        
        # 找到最大的轮廓（假设是手）
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        
        # 如果轮廓面积太小，可能不是手
        if cv2.contourArea(cnt) < 5000:
            return None, None
        
        # 创建凸包
        hull = cv2.convexHull(cnt)
        
        # 计算轮廓的面积和凸包的面积
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        
        # 计算手的特征
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        
        if defects is None:
            return cnt, "Unknown"
        
        # 计算凸缺陷
        n_defects = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            
            # 使用余弦定理计算角度
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
            
            # 如果角度小于90度，认为是手指间的凹陷
            if angle <= np.pi/2:
                n_defects += 1
        
        # 根据凸缺陷数量判断手势
        gesture = "Unknown"
        if n_defects == 0:
            gesture = "Thumbs up"  # 一根手指
        elif n_defects == 4:
            gesture = "Palm"  # 手掌
        elif n_defects == 1:
            gesture = "Point"  # 食指指点
            
        return cnt, gesture
    
    def draw_hand(self, frame, contour, gesture):
        if contour is not None:
            # 绘制手的轮廓
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # 显示手势类型
            if gesture != "Unknown":
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])
                    cv2.putText(frame, f"Gesture: {gesture}", (cx-20, cy-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
