# Face Tracking

一个使用 Python 和 OpenCV 实现的实时人脸跟踪程序。

## 功能特点

- 实时摄像头画面捕获
- 人脸检测和跟踪
- 自动尝试多个摄像头
- 画面水平翻转（镜像效果）
- 详细的错误处理和状态提示

## 环境要求

- Python 3.x
- OpenCV
- NumPy

## 安装

1. 克隆仓库：
```bash
git clone [您的仓库URL]
cd facetrack
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

3. 安装依赖：
```bash
pip install opencv-python numpy
```

## 使用方法

运行程序：
```bash
python face_tracking.py
```

- 按 'q' 键退出程序
- 程序会自动尝试打开可用的摄像头
- 检测到的人脸会用蓝色矩形框标注

## 注意事项

确保已授予程序访问摄像头的权限。在 macOS 中，可能需要在系统偏好设置中允许终端或 VS Code 访问摄像头。
