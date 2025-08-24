# Face and Gesture Tracking

一个使用 Python、OpenCV 和 MediaPipe 实现的实时人脸和手势识别程序。

## 功能特点

- 实时摄像头画面捕获
- 人脸检测和跟踪
- 手势识别和跟踪
  - 支持"指点"手势（Point）
  - 支持"竖大拇指"手势（Thumbs up）
- 手部骨骼点和连接线可视化
- 自动尝试多个摄像头
- 画面水平翻转（镜像效果）
- 详细的错误处理和状态提示

## 环境要求

- Python 3.x
- OpenCV
- NumPy
- MediaPipe

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/ShawnShuoZhang/FaceTrackByClaude.git
cd facetrack
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows 使用：
# .venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install opencv-python numpy mediapipe
```

## 使用方法

运行程序：
```bash
python face_tracking.py
```

### 操作说明

- 按 'q' 键退出程序
- 程序会自动尝试打开可用的摄像头
- 检测到的人脸会用蓝色矩形框标注
- 检测到的手会显示骨骼点和连接线
- 支持的手势：
  - 竖起食指，其他手指弯曲 → 显示 "Point" 手势
  - 竖起大拇指 → 显示 "Thumbs up" 手势

## 注意事项

1. 确保已授予程序访问摄像头的权限：
   - 在 macOS 中：系统偏好设置 → 隐私与安全性 → 摄像头 → 允许终端或 VS Code 访问
   - 在 Windows 中：系统设置 → 隐私 → 相机 → 允许应用访问相机

2. 保持适当的光线条件，这有助于提高识别准确性

3. 手势识别时，确保手部在摄像头视野内，并保持稳定

## 技术实现

- 使用 OpenCV 进行视频捕获和人脸检测
- 使用 MediaPipe 实现手势识别
- 使用 NumPy 进行图像处理

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目！
