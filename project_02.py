import cv2 as cv
import numpy as np


def big_image_binary(image):
    """  
        超大图像二值化
    """
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row + ch, col:col + cw]
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, 127, 10)
            gray[row:row + ch, col:col + cw] = dst
    return gray


def max_area_object_measure(image):
    """  
        找出最大元素，用红线矩形标出来
    """
    binary = big_image_binary(image)
    outimage, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL,
                                                    cv.CHAIN_APPROX_SIMPLE)

    print(type(contours))  #列表数据类型：list
    print()
    max_num = 0
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) > cv.contourArea(contours[max_num]):
            max_num = i

    max_area = cv.contourArea(contours[max_num])  #最大区域的面积
    print("最大对象的面积：%s" % max_area)
    x, y, w, h = cv.boundingRect(contours[max_num])
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv.imwrite('./max_area_object_measure.jpg', image)


# def inside_contour(image):


def cut_picture_roi(image):
    """ 
        裁剪图片的roi：
            选择外围的轮廓进行裁剪
    """
    cnt = findContours(image)  # 返回轮廓
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # cv.circle(image, leftmost, 5, (0, 255, 0), 3)
    # cv.circle(image, rightmost, 5, (0, 0, 255), 3)
    # cv.circle(image, topmost, 5, (0, 0, 255), 3)
    # cv.circle(image, bottommost, 5, (0, 255, 0), 3)

    x1 = min(leftmost[1], rightmost[1], topmost[1], bottommost[1])
    x2 = max(leftmost[1], rightmost[1], topmost[1], bottommost[1])

    y1 = min(leftmost[0], rightmost[0], topmost[0], bottommost[0])
    y2 = max(leftmost[0], rightmost[0], topmost[0], bottommost[0])

    roi = image[x1:x2, y1:y2]  #roi区域
    cv.imwrite("cut.jpg", roi)
    # cv.imshow("roi", image)


def findContours(image):
    gray = big_image_binary(image)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", binary)

    cloneimage, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL,
                                                      cv.CHAIN_APPROX_SIMPLE)

    # for i, contour in enumerate(contours):
    # cv.drawContours(image, contours, i, [0, 0, 255], 2)

    # cv.imwrite("./Detect_Contours.jpg", image)
    return contours[0]  #返回外轮廓


src = cv.imread("./cad3.jpg")
# cv.namedWindow("CAD", cv.WINDOW_AUTOSIZE)
# cv.imshow("CAD", src)
# max_area_object_measure(src)
cut_picture_roi(src)
# findContours(src)

cv.waitKey(0)
cv.destroyAllWindows()
