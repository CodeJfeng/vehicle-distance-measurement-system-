import cv2

from util.onnxTOyolo import Onnx_clf

if __name__ == '__main__':
    clf = Onnx_clf()
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    tk.Tk().withdraw() # 隐藏主窗口, 必须要用，否则会有一个小窗口
    # source = askopenfilename(title="打开保存的图片或视频")
    source = './dis/20240528154848.jpg'
    # source = './data/video/video_20240516_084s151.mp4'
    if source.endswith('.jpg') or source.endswith('.png') or source.endswith('.bmp'):
        res, out = clf.img_identify(source, False , 10, conf = 0.5)
        print(out) # Ture or False
        cv2.imshow('result', res)
        cv2.waitKey(0)
    elif source.endswith('.mp4') or source.endswith('.avi'):
        print('视频识别中...按q退出')
        clf.video_identify(source)
    else:
        print('不支持的文件格式')

