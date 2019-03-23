import cv2 as cv
import numpy as np

page = cv.imread('cad3.jpg')
re_image = cv.resize(page, (np.int(page.shape[1]*0.47), np.int(page.shape[0]*0.47)))
cv.imwrite("cad3.jpg", re_image)

cv.waitKey(0)
cv.destroyAllWindows()