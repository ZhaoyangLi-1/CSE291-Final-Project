from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
from typing import Sequence
import os
import numpy as np

def load_images_from_folder(folder_path):
    """读取指定文件夹内的所有图像文件，并以列表形式返回这些图像，按文件路径的字母顺序排序。

    Args:
        folder_path (str): 图像文件所在的文件夹路径。

    Returns:
        list: 包含PIL图像对象的列表，按文件路径的字母顺序排序。
    """
    images = []  # 创建一个空列表来保存图像
    # 遍历文件夹中的每个文件，先排序
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 检查文件扩展名
            img_path = os.path.join(folder_path, filename)  # 获取图像的完整路径
            try:
                img = Image.open(img_path)  # 使用PIL打开图像
                images.append(img)  # 将图像添加到列表中
            except IOError:
                print(f'Error opening {img_path}')  # 打印无法打开的文件路径
                continue  # 跳过不能打开的文件
    return images


def save_video(frames: Sequence[np.ndarray], filename: str, fps: int = 5):  
    """Save a video composed of a sequence of frames to a file.

    Args:
        frames (Sequence[np.ndarray]): A sequence of image frames as numpy arrays.
        filename (str): The name of the file where the video will be saved.
        fps (int, optional): Frames per second for the output video. Defaults to 5.

    Example:
        frames = [
            controller.step("RotateRight", degrees=5).frame
            for _ in range(72)
        ]
        save_video(frames, 'output_video.mp4', fps=5)  # 这里的fps参数现在是更慢的播放速度的默认值
    """
    frames =  [np.array(image) for image in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, codec='libx264', audio=False)

file_path = "/home/zhaoyang/projects/Ours/alfworld_run/formal-experiment/result/test/scene_5/images"
video_path = os.path.join(file_path, '5.mp4')
frames = load_images_from_folder(file_path)
save_video(frames, video_path, fps=2)