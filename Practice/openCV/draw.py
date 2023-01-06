import cv2 as cv
import numpy as np

blank = np.zeros((500, 500,3), dtype='uint8')
cv.imshow('blank', blank)
# img = cv.imread('/Users/huytuannguyen/Desktop/FPT/My self/Practice/openCV/Screen Shot 2023-01-02 at 10.49.02 PM.png')
# cv.imshow('tuan', img)

# 1. Paint the image a ceratin colour
# blank[:] = 0,255,0
# cv.imshow('green', blank)

# 2. Paint a portion of image a ceratin colour
# blank[200:300, 300:400] = 0,255,0
# cv.imshow('portion', blank)

# # 3. Draw a rectangle
# # cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=-1)   # cv.FILLED == -1    fill all the inside
# cv.rectangle(blank, (0,0), (blank.shape[1] // 2, blank.shape[0] // 2), (0,255,0), thickness=-1)
# cv.imshow('rectangle', blank)

# # 4. Draw circle
# cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 40, (0,0,255), thickness=3)
# cv.imshow('circle', blank)

# # 5. Draw a line
# cv.line(blank, (100,0), (300,400), (255,255,0), thickness=2)
# cv.imshow('line', blank)

# 6. Write text on image
cv.putText(blank, input('dien vao: '), (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
cv.imshow('text', blank)




cv.waitKey(0)