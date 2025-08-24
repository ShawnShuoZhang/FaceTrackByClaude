import cv2
import numpy as np
import time
import mediapipe as mp
import threading
from queue import Queue
from collections import deque
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 创建队列用于线程间通信
frame_queue = Queue(maxsize=2)  # 减小队列大小以降低延迟
result_queue = Queue(maxsize=2)
display_queue = Queue(maxsize=2)

# 帧同步锁
frame_lock = threading.Lock()

class FaceHandTracker:
    def __init__(self):
        # 初始化 MediaPipe（使用 CPU 优化配置）
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # 减少检测手的数量以提高性能
            model_complexity=0,  # 使用最轻量级的模型
            min_detection_confidence=0.3,  # 降低检测置信度阈值
            min_tracking_confidence=0.3
        )
        
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise Exception("无法加载人脸检测模型")
        
        # 性能优化参数
        self.scale_factor = 0.4  # 进一步降低处理分辨率
        self.process_every_n_frames = 2  # 每2帧处理一次，提高性能
        self.frame_count = 0
        self.last_faces = []
        self.last_hand_results = None
        self.flip_horizontal = True  # 控制是否水平翻转
        
        # 预分配内存缓冲区
        self.gray_buffer = None
        self.small_frame_buffer = None
        self.frame_rgb_buffer = None
        
    def init_buffers(self, frame):
        """初始化内存缓冲区"""
        h, w = frame.shape[:2]
        small_h = int(h * self.scale_factor)
        small_w = int(w * self.scale_factor)
        
        self.gray_buffer = np.empty((small_h, small_w), dtype=np.uint8)
        self.small_frame_buffer = np.empty((small_h, small_w, 3), dtype=np.uint8)
        self.frame_rgb_buffer = np.empty((h, w, 3), dtype=np.uint8)
    
    def process_frame(self, frame):
        """处理单个帧"""
        # 初始化缓冲区
        if self.gray_buffer is None:
            self.init_buffers(frame)
        
        self.frame_count += 1
        
        # 根据设置翻转图像
        if self.flip_horizontal:
            cv2.flip(frame, 1, frame)  # 原地翻转以避免内存分配
            
        # 每隔 n 帧进行一次完整处理
        if self.frame_count % self.process_every_n_frames != 0:
            return self.draw_last_results(frame)
        
        # 缩放图像（使用预分配的缓冲区）
        cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, 
                  dst=self.small_frame_buffer)
        
        # 处理人脸检测（使用预分配的缓冲区）
        cv2.cvtColor(self.small_frame_buffer, cv2.COLOR_BGR2GRAY, dst=self.gray_buffer)
        faces = self.face_cascade.detectMultiScale(
            self.gray_buffer,
            scaleFactor=1.2,  # 增加 scaleFactor 以提高性能
            minNeighbors=3,   # 减少 minNeighbors 以提高性能
            minSize=(int(20*self.scale_factor), int(20*self.scale_factor)),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 还原人脸坐标到原始尺寸
        self.last_faces = [(int(x/self.scale_factor), int(y/self.scale_factor),
                           int(w/self.scale_factor), int(h/self.scale_factor))
                          for (x, y, w, h) in faces]
        
        # 处理手势识别（使用预分配的缓冲区）
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self.frame_rgb_buffer)
        self.last_hand_results = self.hands.process(self.frame_rgb_buffer)
        
        return self.draw_last_results(frame)
    
    def draw_last_results(self, frame):
        """绘制最后一次的检测结果"""
        # 绘制人脸检测结果
        for (x, y, w, h) in self.last_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # 绘制手势识别结果
        if self.last_hand_results and self.last_hand_results.multi_hand_landmarks:
            for hand_landmarks in self.last_hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 获取手势信息
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # 检测并显示手势
                gesture = "Unknown"
                if thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y:
                    gesture = "Point"
                elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y:
                    gesture = "Thumbs up"
                
                # 显示识别出的手势
                h, w, _ = frame.shape
                cv2.putText(frame, f"Gesture: {gesture}",
                          (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h - 20)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def capture_frames(cap, frame_queue):
    """捕获视频帧的线程"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 如果队列满了，移除旧的帧
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        
        frame_queue.put(frame)

class FrameBuffer:
    """双缓冲区类"""
    def __init__(self):
        self.buffers = [None, None]
        self.current = 0
    
    def get_current(self):
        return self.buffers[self.current]
    
    def get_next(self):
        return self.buffers[1 - self.current]
    
    def swap(self):
        self.current = 1 - self.current
    
    def initialize(self, shape):
        if self.buffers[0] is None or self.buffers[0].shape != shape:
            self.buffers[0] = np.empty(shape, dtype=np.uint8)
            self.buffers[1] = np.empty(shape, dtype=np.uint8)

def process_frames(tracker, frame_queue, result_queue):
    """处理视频帧的线程"""
    # 设置线程亲和性以提高性能
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([0])  # 将处理线程绑定到第一个CPU核心
    except:
        pass
    
    frame_buffers = FrameBuffer()  # 使用双缓冲
    last_frame_time = time.time()
    frame_interval = 1.0 / 30  # 目标帧间隔（30fps）
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        
        # 初始化或更新缓冲区
        frame_buffers.initialize(frame.shape)
        
        # 复制帧到下一个缓冲区
        np.copyto(frame_buffers.get_next(), frame)
        
        # 处理帧
        with frame_lock:  # 使用锁确保帧处理的原子性
            processed_frame = tracker.process_frame(frame_buffers.get_next())
            frame_buffers.swap()  # 交换缓冲区
        
        # 帧率控制
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_frame_time = time.time()
        
        # 将处理后的帧放入结果队列
        if not result_queue.full():
            result_queue.put(processed_frame)
        else:
            # 如果队列满了，确保使用最新的帧
            try:
                result_queue.get_nowait()
                result_queue.put(processed_frame)
            except:
                pass

def main():
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置稳定的帧率
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 使用最小缓冲区以减少延迟
    
    # 创建显示窗口并设置属性
    cv2.namedWindow('Face and Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Face and Hand Tracking', cv2.WND_PROP_VSYNC, 1)  # 启用垂直同步
    
    # 初始化跟踪器
    try:
        tracker = FaceHandTracker()
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        cap.release()
        return
    
    # 创建线程
    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue,))
    process_thread = threading.Thread(target=process_frames, args=(tracker, frame_queue, result_queue,))
    
    # 启动线程
    capture_thread.start()
    process_thread.start()
    
    # FPS 计算
    fps_time = time.time()
    fps_frames = 0
    fps = 0
    
    print("程序已启动，按'q'键退出")
    
    try:
        while True:
            # 获取处理后的帧
            if not result_queue.empty():
                frame = result_queue.get()
                
                # 计算 FPS
                fps_frames += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_frames / (time.time() - fps_time)
                    fps_frames = 0
                    fps_time = time.time()
                
                # 显示 FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('Face and Hand Tracking', frame)
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("正在关闭程序...")
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 停止线程
        frame_queue.put(None)
        result_queue.put(None)
        capture_thread.join()
        process_thread.join()

if __name__ == '__main__':
    main()
