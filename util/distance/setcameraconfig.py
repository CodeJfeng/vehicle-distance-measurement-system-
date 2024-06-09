import numpy as np


class cameraConfig:
    def __init__(self):
        # 相机内参
        self.cam_matrix_left = np.array([[399.977814879486, 0, 84.0892986579545],
                                        [0, 7.591605137647632e+02, 248.310763279748],
                                        [0, 0, 1]])
        self.cam_matrix_right = np.array([[398.049798237015, 0, 82.6482275576469],
                                          [0, 761.717360135140, 258.731184679570],
                                          [0, 0, 1]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.178885424463741, -0.210837351592580, 0.00403416172715268, -0.0687818068465347, 0]])
        self.distortion_r = np.array([[0.157799863672156, -0.218711923677998, 0.00642308784630767, -0.0677048249585232, 0]])
        # 旋转矩阵
        self.R = np.array([[0.999766570045367, 0.00178668314283566, 0.0215316785938505],
                            [-0.00153513063622705, 0.999930447093110, -0.0116937739888544],
                            [-0.0215510740718761, 0.0116579903722639, 0.999699776166239]])
        # 平移矩阵
        self.T = np.array([[-40.6946564625264], [0.202465466675109], [6.54704922256347]])
        # 主点列坐标的差
        self.doffs = 0.0
        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    def setMiddleComfig(self):
        # 平均相机内参
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True

    def getConfig(self):
        return {'cam_matrix_left':self.cam_matrix_left,
                'cam_matrix_right':self.cam_matrix_right,
                'distortion_l':self.distortion_l,
                'distortion_r':self.distortion_r,
                'R':self.R, 'T':self.T,
                'doffs':self.doffs, 'isRectified': self.isRectified}