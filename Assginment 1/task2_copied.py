import cv2 as cv
import numpy as np
import time
from scipy import stats

####################################################################################################################
alpha = 0.001           # learning rate
initWt = 0.001          # weight of a new gaussian
initVar = 255           # variance of a new gaussian
maxG = 5                # maximum number of gaussians
wtsThreshold = 0.9      # weight threshold for background

####################################################################################################################
def work(inputVec):
    bigDataIn = inputVec.reshape(4 + 5 * maxG)
    pixData = bigDataIn[0:3]
    frameStrut = bigDataIn[3:]
    flagModelFit = False
    
    for k in range(int(frameStrut[-1])):
        dis = pixData - frameStrut[2 * maxG + 3 * k:2 * maxG + 3 * (k + 1)]
        
        if np.dot(dis, dis) < 6.25 * frameStrut[maxG + k]:
            flagModelFit = True
            rho = alpha * stats.multivariate_normal.pdf(
                pixData, frameStrut[2 * maxG + 3 * k:2 * maxG + 3 * (k + 1)], frameStrut[maxG + k] * np.eye(3))
            frameStrut[k] = (1 - alpha) * frameStrut[k] + alpha
            frameStrut[maxG + k] = (1 - rho) * frameStrut[maxG + k] + rho * np.dot(dis, dis)
            frameStrut[2 * maxG + 3 * k:2 * maxG + 3 * (k + 1)] = (1 - rho) * frameStrut[2 * maxG + 3 * k:2 * maxG + 3 * (k + 1)] + rho * pixData
            break
        else:
            frameStrut[k] = (1 - alpha) * frameStrut[k]
    
    if not flagModelFit:
        if frameStrut[-1] < maxG:
            ind = int(frameStrut[-1])
            frameStrut[-1] += 1
        else:
            ind = np.argmin(np.divide(frameStrut[0:maxG], frameStrut[maxG:2 * maxG]))
        
        frameStrut[ind] = initWt
        frameStrut[maxG + ind] = initVar
        frameStrut[2 * maxG + 3 * ind:2 * maxG + 3 * (ind + 1)] = pixData
    
    frameStrut[0:maxG] = np.divide(frameStrut[0:maxG], np.sum(frameStrut[0:maxG]))
    
    indexing = np.argsort(np.divide(frameStrut[0:int(frameStrut[-1])], frameStrut[maxG:maxG + int(frameStrut[-1])]))
    wtSum = 0
    for k in indexing[::-1]:
        dis = pixData - frameStrut[2 * maxG + 3 * k:2 * maxG + 3 * (k + 1)]
        var = frameStrut[maxG + k]
        if np.dot(dis, dis) < 6.25 * var:
            pixData[:] = 255 * np.ones(3)
        wtSum += frameStrut[k]
        if wtSum > wtsThreshold:
            break
    return np.concatenate((pixData, frameStrut))

####################################################################################################################
cap = cv.VideoCapture('video.mpeg')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

wd = int(cap.get(3))  # width of frame
ht = int(cap.get(4))  # height of frame
if wd == 0 or ht == 0:
    print("Error: Invalid video dimensions.")
    exit()

new_width = 240
new_height = int((new_width / wd) * ht)

print(f"Video loaded successfully: Width={wd}, Height={ht}, Resized Width={new_width}, Resized Height={new_height}")

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (new_width, new_height))
if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

bigDataMat = np.zeros((new_height, new_width, 4 + 5 * maxG))
tStart = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
        break
    
    frame = cv.resize(frame, (new_width, new_height))
    tFrameStart = time.time()
    bigDataMat[:, :, 0:3] = frame
    inputData = [bigDataMat[i, j, :].reshape(4 + 5 * maxG) for i in range(new_height) for j in range(new_width)]
    
    data = [work(vec) for vec in inputData]
    bigDataMat = np.reshape(np.concatenate(data, axis=0), (new_height, new_width, 4 + 5 * maxG))
    
    editFrame = np.uint8(bigDataMat[:, :, 0:3])
    print('Last frame took : {:3.2f} secs'.format(time.time() - tFrameStart))
    
    cv.imshow('Frame', frame)
    cv.imshow('Editedframe', editFrame)
    out.write(editFrame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
print('Total processing time was {0:0.2f} sec'.format(time.time() - tStart))
