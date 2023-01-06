import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    # work for image, video, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # work for live video
    capture.set(3, width)
    capture.set(4, height)

# rescale image

img = cv.imread('/Users/huytuannguyen/Desktop/FPT/My self/Practice/openCV/cat_large.jpeg')
rescaled_image = rescaleFrame(img)
cv.imshow('image', img)
cv.imshow('resized image', rescaled_image)

cv.waitKey(0)



# rescale video

capture = cv.VideoCapture('/Users/huytuannguyen/Desktop/FPT/My self/Practice/openCV/VID_20230106_093117_879.mp4')
while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('video', frame)
    cv.imshow('video resized', frame_resized)

    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()