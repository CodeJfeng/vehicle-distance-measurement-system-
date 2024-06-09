import os



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


if __name__ == "__main__":
    label_to_num()
    pass

