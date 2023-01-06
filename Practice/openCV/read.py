import cv2 as cv

# read image

# img = cv.imread('/Users/huytuannguyen/Desktop/FPT/My self/Practice/openCV/Screen Shot 2023-01-02 at 10.49.02 PM.png')
# cv.imshow('tuan', img)
# cv.waitKey(0)


# read video

capture = cv.VideoCapture('/Users/huytuannguyen/Desktop/FPT/My self/Practice/openCV/VID_20230106_093117_879.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()



