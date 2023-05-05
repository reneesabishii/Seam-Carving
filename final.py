import cv2
import numpy as np
import logging
import argparse
import sys

#crop image, change to size of both seams removed and compare
def crop(img, pixels):
    rows, cols = img.shape[:2]
    startR = int(0 + (pixels/2))
    endR = int(rows - (pixels/2))
    startC = int(0 + (pixels/2))
    endC = int(cols - (pixels/2))
    croppedImg = img[startR:endR, startC:endC]
    cv2.imwrite("Cropped.jpg", croppedImg) 

def findSeam(energy):
    rows, cols = energy.shape[:2]
    M = []
    M.append(energy[0].tolist())
    #i = second row
    #M(i, j) = e(i, j)+ min(M(i−1, j −1),M(i−1, j),M(i−1, j +1))
    #to last the last row
    for i in range(1, rows):
        newM = []
        for j in range(cols):
            if j == 0:
                newM.append(energy[i][j] + min(M[i-1][j], M[i-1][j+1]))
            elif j == cols - 1:
                newM.append(energy[i][j] + min(M[i-1][j], M[i-1][j-1]))
            else:
                #if not edge then can be all three
                newM.append(energy[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1]))
        M.append(newM)

    #get full seams
    seam = [0] * rows
    seam[rows-1] = np.argmin(M[rows-1])
    for i in range(rows-2, -1, -1):
        j = seam[i+1]
        if j == 0:
            seam[i] = np.argmin(M[i][j:j+2]) + j
        elif j == cols - 1:
            seam[i] = np.argmin(M[i][j-1:j+1]) + j-1
        else:
            seam[i] = np.argmin(M[i][j-1:j+2]) + j-1
    return seam

def getEnergy(img):
    #first convert image to gray scale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Sobel operator Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT); 0, 1 for y
    #X and Y derivative of the image using openCV, 3x3 sobel kernel
    sobelX = cv2.Sobel(grayImg,cv2.CV_64F, 1, 0, ksize=3) 
    sobelY = cv2.Sobel(grayImg,cv2.CV_64F, 0, 1, ksize=3) 
    sobelX = cv2.convertScaleAbs(sobelX) 
    sobelY = cv2.convertScaleAbs(sobelY) 
 
    #return with weighted approximation
    energy = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    return energy

#remove seam from image
def removeSeam(img, seam):
    rows, cols = img.shape[:2] 
    for row in range(rows): 
        for col in range(int(seam[row]), cols-1): 
            img[row, col] = img[row, col+1] 
    img = img[:, 0:cols-1] 
    return img 

#add a seam to the image 
def addSeam(img, seam): 
    rows, cols = img.shape[:2] 
    zeroCol = np.zeros((rows,1,3), dtype=np.uint8) 
    imgAdd = np.hstack((img, zeroCol)) 
    for row in range(rows): 
        for col in range(cols, int(seam[row]), -1): 
            imgAdd[row, col] = img[row, col-1] 
    return imgAdd

#overlay the seams
def overlaySeam(img, seam): 
    imgOverlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i,int(j)) for i,j in enumerate(seam)])  
    imgOverlay[x_coords, y_coords] = (128,0,255) 
    return imgOverlay  

def rotateImg(img, x):
    if x == 0:
        rImg = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        rImg = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rImg
if __name__ == "__main__":
    #input modeled after assignments
    #python3 final.py --crop test.jpg --pixels 10
    #example for seam carving python3 final.py --seamcarving test.jpg --seams 10
    #The default amount of seams/pixels is 10
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Choose standard: "crop" or "seam carving"')
    parser.add_argument('--crop',
                        help='Use opencv crop')
    parser.add_argument('--seamcarving',
                        help='Use seam carving')
    parser.add_argument('--seams', type=int,
                        help='how many seams you want to add and remove', default=10)
    parser.add_argument('--pixels', type=int,
                        help='number of pixels to crop', default=10)
    args = parser.parse_args()

    imgInput = cv2.imread(sys.argv[2])

    if args.crop:
        pixels = int(args.pixels)
        crop(imgInput, pixels)
    
    if args.seamcarving:
        numSeams = int(args.seams) 

        imgV = np.copy(imgInput)
        imgAddV = np.copy(imgV)
        imgH = np.copy(rotateImg(imgInput, 0))
        imgAddH = np.copy(imgH)
        imgB = np.copy(imgInput)
        imgAddB = np.copy(imgInput)

        imgOverlayV = np.copy(imgV)
        imgOverlayH = np.copy(imgH)
        imgSeamBoth = np.copy(imgB)

        for i in range(numSeams):
            energyMatrixV = getEnergy(imgV) 
            energyMatrixH = getEnergy(imgH)
            energyMatrixAddH = getEnergy(imgAddH)
            energyMatrixAddV = getEnergy(imgAddV)
            energyMatrixB = getEnergy(imgB)
            energyMatrixAddB = getEnergy(imgAddB)
            vSeam = findSeam(energyMatrixV)
            imgOverlayV = overlaySeam(imgOverlayV, vSeam)
            imgSeamBoth = overlaySeam(imgSeamBoth, vSeam)

            imgV = removeSeam(imgV, vSeam)
            imgAddV = addSeam(imgAddV, vSeam)

            #find horizontal seam
            hSeam = findSeam(energyMatrixH)
            imgOverlayH = overlaySeam(imgOverlayH, hSeam)

            imgSeamBoth = rotateImg(imgSeamBoth, 0)
            imgSeamBoth = overlaySeam(imgSeamBoth, hSeam)
            imgSeamBoth = rotateImg(imgSeamBoth, 1)

            imgH = removeSeam(imgH, hSeam)
            imgAddH = addSeam(imgAddH, hSeam)

            imgB = removeSeam(imgB, vSeam)
            imgAddB = rotateImg(imgAddB, 0)
            imgB = rotateImg(imgB, 0)
            imgB = removeSeam(imgB, hSeam)
            imgB = rotateImg(imgB, 1)
            imgAddB = rotateImg(imgAddB, 1)

        imageHorizontalOverlay = rotateImg(imgOverlayH, 1)
        imgH = rotateImg(imgH, 1)
        imgAddH = rotateImg(imgAddH, 1)
         
        cv2.imwrite("1Input.jpg", imgInput)
        cv2.imwrite('2SeamsV.jpg', imgOverlayV)
        cv2.imwrite('3SeamsH.jpg', imageHorizontalOverlay)
        cv2.imwrite('4SeamsB.jpg', imgSeamBoth)    
        cv2.imwrite('5OutputV.jpg', imgV) 
        cv2.imwrite('6OutputH.jpg', imgH)
        cv2.imwrite('7OutputB.jpg', imgB)
        cv2.imwrite('8OutputaddH.jpg', imgAddH)
        cv2.imwrite('9OutputaddV.jpg', imgAddV)




