#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image

yolo = YOLO() # YOLO类

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image) # 检测图片
        r_image.show()
