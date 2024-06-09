import os

width = 640
height = 384
classes = ['car', 'truck', 'bus', 'pedestrian', 'bicycle','motorcycle','tricycle']

def convert(box):
    x = box[0] / width
    y = box[1] / height
    w = box[2] / width
    h = box[3] / height

    return [str(x), str(y), str(w), str(h)]

def covert_txt(file_path):
    content = ''
    with(open(file_path, 'r')) as fp:
        for f in fp.readlines():
            f = f.split(" ")
            bb = [float(f[1]),float(f[2]),float(f[3]),float(f[4])]
            content = content+f[0]+' '+' '.join(convert(bb))+'\n'
    file = open(file_path, 'w')
    file.write(content)
    file.close()

def label_to_num():
    path = r'./labels/'
    files = os.listdir(path)
    for file in files:
        print(file)
        f = open(path+file,'r')
        contend = f.read()
        for i in range(0,len(classes)):
            new_content = contend.replace(classes[i],str(i))
            contend = new_content
        f.close()
        f = open(path+file, 'w')
        f.write(new_content)
        f.close()

if __name__ == '__main__':
    path = r'./labels/'
    for file in os.listdir(path):
        covert_txt(path+file)
    pass