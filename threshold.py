import cv2 as cv
import numpy as np
from PIL import Image as PilImg
import pytesseract
import os
import sys


def threshold_function(image):
    """ 
        图像二值化：
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    print("阈值：%s" % ret)
    cv.imshow("binary", binary)


def big_image_binary(image):
    """  
        超大图像二值化
    """
    cw = 256
    ch = 256
    h, w = image.shape[:2]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):  #0~h,步长为256
        for col in range(0, w, cw):  #0~w,步长为256
            roi = gray[row:row + ch, col:col + cw]  #roi区域
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, 127, 10)  #分块局部阈值
            gray[row:row + ch, col:col + cw] = dst
            # print(np.std(dst), np.mean(dst))
    # cv.imshow("Big Binary", gray)
    # cv.imwrite('./binary.jpg',gray)
    return gray


def open_function(image):
    """  
        开操作
    """
    binary = big_image_binary(image)  #二值化
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))  #结构元素，去掉水平线
    dst = cv.morphologyEx(binary, op=cv.MORPH_OPEN, kernel=kernel)
    # cv.imshow("OPEN",dst)
    cv.imwrite('./open.jpg', dst)


def object_measure(image):
    """  
        对象测量
    """
    binary = big_image_binary(image)
    outImage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL,
                                                   cv.CHAIN_APPROX_SIMPLE)
    
    print(type(contours))
    print( enumerate(contours))
    print(cv.contourArea(contours[1]))
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        mm = cv.moments(contour)
        # cx = mm['m10']/mm['m00']
        # cy = mm['m01']/mm['m00']
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imwrite('./object_measure.jpg', image)


def ocr_number(image):
    """  
        ocr识别数字的区域，然后设置为ROI
    """
    img = PilImg.open('./cad1.jpg')
    text = pytesseract.image_to_string(img)
    # print(text)
    text = text.replace('\n','')

    txt_path = './ocr.txt'
    if os.path.exists(txt_path) == False:
        txt = open(txt_path,'w') #新建txt文件
    else:
        txt = open(txt_path,'a') #不会清空原来txt文件
    txt.write(text)
    txt.write("\n\n")
    txt.close()



src = cv.imread("./cad1.jpg")
# cv.namedWindow("cad", cv.WINDOW_AUTOSIZE)
# cv.imshow("cad", src)
object_measure(src)
# open_function(src)
# ocr_number(src)

cv.waitKey(0)
cv.destroyAllWindows()
