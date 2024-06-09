# import cv2
#
#
# def onMouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#
# pre_pixel = [1080, 900, 760, 696, 657, 617, 596, 567, 554, 533, 522, 499, 479, 468, 427, 401, 390]
# image = cv2.imread('./data/images/20240515091357.jpg')
# for p in pre_pixel:
#     cv2.line(image, (0,p), (1920, p),(0,0,255))
#
# cv2.imshow('test', image)
# cv2.setMouseCallback("test", onMouse, 0)
# cv2.waitKey(0)


import math


def calculate_speed(coord1, coord2, time_interval):
    """
    计算车速
    coord1: 第一个坐标点 (lat1, long1, time1)
    coord2: 第二个坐标点 (lat2, long2, time2)
    time_interval: 两个坐标点的时间间隔（秒）
    """
    (lat1, long1, time1) = coord1
    (lat2, long2, time2) = coord2

    # 地球半径（km）
    earth_radius = 6371.0

    # 转换纬度和经度到弧度
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)

    # 经纬度差值
    d_lat = math.radians(lat2 - lat1)
    d_long = math.radians(long2 - long1)

    # 计算两点之间的距离
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    # 计算速度（km/h）
    speed = distance / (time_interval / 3600 )   # 单位转换1h = 3600s

    return speed


# 使用函数的例子
# 假设起点坐标和时间
start_coord = (34.750555, 113.600555, 1616750400)  # (lat, long, time)
# 终点坐标和时间
end_coord = (34.7508333, 113.609444, 1616750410)  # (lat, long, time)

# 时间间隔是从起点到终点的秒数
time_interval = (end_coord[2] - start_coord[2])

# 计算速度
speed = calculate_speed(start_coord, end_coord, time_interval)
print(f"Speed: {speed} km/h")