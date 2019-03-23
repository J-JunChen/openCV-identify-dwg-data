import cv2 as cv
import numpy as np
import json
import math


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
    cv.imwrite("binary.jpg", gray)
    return gray


def max_area_object_measure(image):
    """  
        找出最大元素，用红线矩形标出来
    """
    binary = big_image_binary(image)
    outimage, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL,
                                                    cv.CHAIN_APPROX_SIMPLE)

    print(type(contours))  # 列表数据类型：list
    max_num = 0
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) > cv.contourArea(contours[max_num]):
            max_num = i

    max_area = cv.contourArea(contours[max_num])  # 最大区域的面积
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
    cnt = max_roi(image)  # 返回轮廓

    # leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    # rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    # topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    # bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # cv.circle(image, leftmost, 5, (0, 255, 0), 3)
    # cv.circle(image, rightmost, 5, (0, 0, 255), 3)
    # cv.circle(image, topmost, 5, (0, 0, 255), 3)
    # cv.circle(image, bottommost, 5, (0, 255, 0), 3)

    # x1 = min(leftmost[1], rightmost[1], topmost[1], bottommost[1])
    # x2 = max(leftmost[1], rightmost[1], topmost[1], bottommost[1])
    # y1 = min(leftmost[0], rightmost[0], topmost[0], bottommost[0])
    # y2 = max(leftmost[0], rightmost[0], topmost[0], bottommost[0])
    # roi = image[x1:x2, y1:y2]  #roi区域
    # cv.imwrite("cut.jpg", roi)

    x, y, w, h = cv.boundingRect(cnt)
    # cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = image[y:y + h, x:x + w]
    cv.imwrite("cut.jpg", roi)

    # cv.imshow("roi", image)


def max_roi(image):
    gray = big_image_binary(image)
    dst = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    cloneimage, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE,
                                                      cv.CHAIN_APPROX_SIMPLE)

    return contours[0]


def find_contours(image):
    gray = big_image_binary(image)
    dst = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", dst)

    # cloneimage, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL,
    #                                                   cv.CHAIN_APPROX_SIMPLE)

    cloneimage, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE,
                                                      cv.CHAIN_APPROX_SIMPLE)
    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]

    # for i, contour in enumerate(contours):
    #     cv.drawContours(image, contours, i, [0, 0, 255], 2)
    # cv.imwrite("./image.jpg", image)

    room_num = input("请用户输入多少个房间：")

    # len(contours)
    contour_area = []
    for contour in contours:
        contour_area.append(cv.contourArea(contour))
    contour_area = quick_sort(contour_area)

    room_contours = []  # 用户指定房间的

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv.boundingRect(currentContour)
        if currentHierarchy[2] < 0:
            # these are the innermost child components
            # cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            for i in range(1, int(room_num)+1):  # 因为最大的区域就是整一张图纸
                if cv.contourArea(currentContour) == contour_area[i]:
                    print("area：%s" % cv.contourArea(currentContour))
                    cv.drawContours(dst, [currentContour],
                                    0, (214, 238, 247), -1)
                    room_contours.append(currentContour)

        elif currentHierarchy[3] < 0:
            # these are the outermost parent components
            cv.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 0), 1)

    cv.imwrite("./Detect_Contours.jpg", dst)

    """  抽取其中一个房间标出其中一条边，两个点 """
    # x0, y0, w0, h0 = cv.boundingRect(room_contours[0])
    # cv.circle(dst, (x0, y0), 5, (0, 0, 255), 4)
    # cv.circle(dst, (x0+w0, y0), 5, (0, 0, 255), 4)
    # cv.line(dst, (x0, y0), (x0+w0, y0), (0, 0, 255), 3)
    # height, width = map(float, input("请分别输入标定房间的实际高度， 宽度：").split(','))
    # print("高度为：%d" % height + ", 宽度：%d" % width)

    save_contours(dst, room_contours)


def quick_sort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            else:
                greater.append(x)
        # Don't forget to return something!
        # Just use the + operator to join lists
        return quick_sort(greater)+equal+quick_sort(less)
    # Note that you want equal ^^^^^ not pivot
    # You need to hande the part at the end of the recursion - when you only have one element in your array, just return the array.
    else:
        return array


def save_contours(img, contours):
    height, width = img.shape[:2]

    num = 0
    for contour in contours:

        cv.drawContours(img, [contour],
                        0, (255, 255, 255), -1)
        for i in range(0,  height):
            for j in range(0, width):
                if cv.pointPolygonTest(contour, (j, i), False) < 0:
                    img[i, j] = [0, 0, 0]

        x, y, w, h = cv.boundingRect(contour)
        print("width：%d" % w + ", height : %d" % h)
        # cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

        """ 画坐标系 """
        # cv.circle(img, (x, y+h), 5, (0, 0, 255), 4)
        # cv.arrowedLine(img, (x, y+h), (x, y+h-1003), (255, 0, 0), 3)
        # cv.arrowedLine(img, (x, y+h), (x+100, y+h), (255, 0, 0), 3)

        cv.imwrite("save_contour_" + str(num) + ".jpg", img[y:y+h, x:x+w])
        num += 1


def rotate_picture(img):
    """ 旋转图片 """
    rotate_direction = input("左转还是右转? R/L：")

    if rotate_direction.lower() == 'l':
        left_rotated = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        cv.imwrite("./左转90度.jpg", left_rotated)

    else:
        right_rotated = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite("./右转90度.jpg", right_rotated)


def line_detect(img):
    edge = canny_edge_function(img)

    '''
        cv.HoughLinesP（常用）:返回准确位置的起始点和终止点，一个一个细小的线段，
        则可以根据线段进行测距
            rho：半径步长
            theta：角度，每次偏转一度
            threshold：自定义低阈值
            minLineLength：最小线的长度为50个像素，小于50就不算是一个线段。
            maxLineGap = 10：同一直线上两个线段的间隙距离小于10的话，就把两个线段连接起来，当做一条线段
    '''
    lines = cv.HoughLinesP(edge, rho=1, theta=np.pi/180,
                           threshold=100, minLineLength=0, maxLineGap=10)

    line_len = []
    # copy_img = []
    copy_img = np.zeros(img.shape, np.uint8)  # 根据图像的大小来创建一个图像对象

    # cv.imshow("emptyImage", copy_img)

    for i in range(len(lines)):
        copy_img = img.copy()

        x1, y1, x2, y2 = lines[i][0]
        cv.line(copy_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv.imwrite("请输入长度" + str(i)+'.jpg', copy_img)
        # l= input("请输入显示边的实际长度：")
        # line_len.append(int(l))

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv.imwrite("./直线检测.jpg", img)


def canny_edge_function(image):
    '''
        Canny边缘提取算法：
            1、进行高斯模糊：因为Canny对噪声比较敏感，所以先高斯模糊降噪
            2、灰度转移：转化为单通道
            3、计算梯度：Sobel/Scharr
            4、
            5、高低阈值输出二值图像
    '''
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # 低阈值：50；高阈值：150
    edge_output = cv.Canny(grad_x, grad_y, 50, 150)
    cv.imshow("Binary", edge_output)
    return edge_output


def point_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    binary = canny_edge_function(img)
    cloneImage, contours, hierachy = cv.findContours(
        binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(img, contours, i, (0, 255, 0), 1)
        print(i)
    cv.imshow("Detect_contours", img)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)
    img[dst > 0.01*dst.max()] = [0, 0, 255]
    cv.imshow("角点检测", img)

    height, width = img.shape[:2]

    num = 0
    points = []

    for row in range(height-1,  -1, -4):
        for col in range(0, width, 4):
            # print("BGR：", img[j, i][0])
            pixel = img[row, col]
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
                # img[i, j] = [0, 0, 0]
                cv.circle(img, (col, row), 5, (0, 255, 0), 5)
                cv.imwrite("./点图" + str(num) + '.jpg', img)
                print("row_%d" % num + "：%d" %
                      row + ", col_%d " % num + "：%d" % col)
                # points.append({"row": row, "col": col})
                points.append([np.abs(height - row) , col])
                num += 1

    points = np.asarray(points) # 将 list 转换成 array
    print(points)
    estimate_rectangle(points, num)


def estimate_rectangle(points, pnum):
    """ 根据四个点的坐标，判断是否矩形 """
    if pnum == 4 and points[0]["col"] == points[2]['col'] and points[0]['row'] == points[1]['row'] and points[1]['col'] == points[3]['col']:
        print("是矩形")
        print("请输入长和宽")
    else:
        print("不是矩形")
        print("请从坐标原点开始，顺时针输入各个点的坐标")
    adjacency_matrix(points, pnum)


def adjacency_matrix(points, pnum):
    """ 邻接矩阵 """
    adjacency = np.zeros((pnum, pnum), dtype=np.int8)  # 邻接矩阵
    print(adjacency)

    """ 判断角点之间是否相连 """
    # for i in range(pnum):
    #     for j in range(i+1, pnum):

    # 下面就先假设邻接矩阵已经完成
    adjacency[0, 1] = adjacency[0, 6] = adjacency[1, 0] = adjacency[6, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1
    adjacency[3, 4] = adjacency[4, 3] = 1
    adjacency[4, 5] = adjacency[5, 4] = 1
    adjacency[5, 7] = adjacency[7, 5] = 1
    adjacency[6, 7] = adjacency[7, 6] = 1
    print(adjacency)

    clockwise = [0] # 顺时针排序后的点，第一点为第0点
    one_array = np.where(adjacency[0] == 1)[0]
    print(one_array)
    
   # print(angle3pt(points[one_array[0]], points[0], points[one_array[1]]))

    sort_clockwise(adjacency, points, one_array, 0 , clockwise, pnum) # 第0行开始
    print(clockwise)

def sort_clockwise(adjacency, points, one_array, i, clockwise, pnum):
    """ 
    递归：
        角点顺时针排序算法，基于邻接矩阵的遍历算法:
        1、第一步，先找出与第0点的两个点（即第1点和第6点），哪个是顺时针最先连接的。
        2、第二步：找到顺时针遍历的第二个点之后（即第6点），就根据邻接矩阵的遍历算法对接下来的角点进行排序
        3、第三步：判断修改后的邻接矩阵每行中，如果“1”的个数为2，则回到第二步，否则为“1”就是顺时针的下一个点
    """
    k = 0
    # 第三步：判断遍历行”1“的个数，如果只有每行中少于2点为“1”的，即该行的下一个连接点就是列中为“1”的点
    if len(clockwise) == pnum: 
        return # 递归结束条件
    elif len(one_array) == 1: 
        k = one_array[0]
    else: # 除非就有两个点，则执行第一步
        if angle3pt(points[one_array[0]], points[i], points[one_array[1]]) >=0 :
            """ 如果两个向量之间的角度大于0，即one_array[0]为顺时针的第一个点 """
            k = one_array[0]
        else:
            k = one_array[1]
    
    adjacency[i, k] = adjacency[k, i] = 0
    clockwise.append(k)

    one_array = np.where(adjacency[k] == 1)[0]
    sort_clockwise(adjacency, points, one_array, k, clockwise, pnum)


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    # return ang + 360 if ang < 0 else ang
    return ang


        


# src = cv.imread("./cad3.jpg")
# cv.namedWindow("CAD", cv.WINDOW_AUTOSIZE)
# cv.imshow("CAD", src)
# max_area_object_measure(src)
# cut_picture_roi(src)
# src = cv.imread("./cut.jpg")
# find_contours(src)
src = cv.imread('./左转90度.jpg')
# rotate_picture(src)
# line_detect(src)
point_detection(src)


cv.waitKey(0)
cv.destroyAllWindows()
