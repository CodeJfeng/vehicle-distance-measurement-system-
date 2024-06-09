import numpy as np


class danmu:
    def __init__(self):  # 可以用于标定相机参数
        self.high = 1.70
        self.pre_pixel = [1080, 865, 805, 760, 690, 655, 625, 600, 570, 550, 536, 523, 499, 480, 460, 440, 422, 402, 380, 360, 340, 325, 310]
        self.pre_dist = [0, 0.5, 1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,44]

    def mydistance(self, image, boxs):  # boxs -> [x1,y1, x2, y2] 计算几步
        h,w,_ = image.shape # 默认为 1920 * 1080
        dist = np.zeros(len(boxs))
        for x, box in enumerate(boxs):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, y2)
            for i, p in enumerate(self.pre_pixel):
                if p <= center[1]:
                    if i == 0:
                        dist[x] = 0
                    else:
                        dist_top = self.pre_dist[i]
                        dist_bottom = self.pre_dist[i - 1]
                        piexl_top = self.pre_pixel[i]
                        piexl_bottom = self.pre_pixel[i - 1]
                        piexl_sum = piexl_bottom - piexl_top
                        dist_sum = dist_top - dist_bottom
                        nomr = dist_sum / piexl_sum
                        dist[x] = dist_top - nomr * (p - center[1])
                    hang = abs(center[0] - w/2) / abs(h - center[1]) * dist[x]
                    dist[x] = (dist[x] **2 + hang ** 2) ** 0.5
        return dist


if __name__ == '__main__':
    danmu = danmu()
    danmu.distance(None, [[1, 1, 1, 1]])
