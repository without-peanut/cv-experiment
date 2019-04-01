# -*- coding: utf-8 -*-
import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.
       这是一个有用的辅助函数，您可以选择它来实现一个图像和一个转换，并计算转换后的图像的边界框。

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")
    h = img.shape[0] - 1
    w = img.shape[1] - 1

    a1 = np.array([[0, 0, 1]]).T
    b1 = np.array([[w, 0, 1]]).T
    c1 = np.array([[0, h, 1]]).T
    d1 = np.array([[w, h, 1]]).T

    a2 = np.dot(M, a1)
    b2 = np.dot(M, b1)
    c2 = np.dot(M, c1)
    d2 = np.dot(M, d1)

    a2 /= a2[-1]
    b2 /= b2[-1]
    c2 /= c2[-1]
    d2 /= d2[-1]

    minX = min(a2[0], b2[0], c2[0], d2[0])
    minY = min(a2[1], b2[1], c2[1], d2[1])
    maxX = max(a2[0], b2[0], c2[0], d2[0])
    maxY = max(a2[1], b2[1], c2[1], d2[1])

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
         输入:
         img:要添加到累加器中的图像
         acc:累积图像中需要添加img的部分
         M:将输入图像映射到累加器的变换
         混合宽度:混合函数的宽度。水平的帽子函数
         输出:
         增加img的加权拷贝对acc进行修改，acc的前三个通道记录像素颜色的加权和，acc的第四个通道记录权重的和
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    #raise Exception("TODO in blend.py not implemented")

    x1, y1, x2, y2 = imageBoundingBox(img, M)
    row_num = y2-y1
    col_num = x2-x1
    x_range = np.arange(x1, x2)
    y_range = np.arange(y1, y2)
    (x_mesh, y_mesh) = np.meshgrid(x_range, y_range)
    one = np.ones((row_num, col_num))
    allLoc = np.dstack((x_mesh, y_mesh, one))
    allLoc = allLoc.reshape((row_num*col_num, 3))
    allLoc = allLoc.T
    loc2 = np.dot(np.linalg.inv(M), allLoc)  # 卷绕到了原来的坐标
    loc2 = loc2 / loc2[2]  # z 设为 1

    map_x = loc2[0].reshape((row_num, col_num)).astype(np.float32)  # 挑出来 做 个类型转换
    map_y = loc2[1].reshape((row_num, col_num)).astype(np.float32)

    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    weight = np.ones((row_num, col_num))

    blendc = np.arange(0+1.0/blendWidth, 1+1.0/blendWidth, 1.0/blendWidth)
    weight[:, 0:blendWidth] = blendc
    weight[:, col_num-blendWidth:col_num] = -np.sort(-blendc)
    acc1 = np.zeros((row_num, col_num,3))
    acc1 += np.multiply(dst, weight.reshape(row_num, col_num, 1))
    #acc[y1:y2, x1:x2, 0:3] += np.multiply(dst, weight.reshape(row_num, col_num, 1))

    weightc = acc1.sum(axis=2)
    weightc = (weightc > 0)*1.0
    weight = np.multiply(weight, weightc)

    '''
    for i in range(x2-x1):
        for j in range(y2-y1):
            if (acc1[j, i] - [0, 0, 0]).any() == False:
                    weight[j, i] = 0.0
    '''
    acc[y1:y2, x1:x2, 0:3] += acc1
    acc[y1:y2, x1:x2, 3] += weight





    '''
    for i in range(x1, x2):
        for j in range(y1, y2):
            a = np.array([[i, j, 1]],dtype = float).T
            b = np.dot(np.linalg.inv(M), a)
            b /= b[-1]
            x = b[0]
            y = b[1]

            if 0 <= x <= w and 0 <= y <= h:
                if x1 <= i <= x1 + blendWidth:
                    weight = float(i - x1) / blendWidth

                elif x2 - blendWidth <= i <= x2:
                    weight = float(x2 - i) / blendWidth

                else:
                    weight = 1.0

                img1 = img[int(math.floor(y)), int(math.floor(x))]
                img2 = img[int(math.floor(y)), int(math.ceil(x))]
                img3 = img[int(math.ceil(y)), int(math.floor(x))]
                img4 = img[int(math.ceil(y)), int(math.ceil(x))]

                img5 = (math.ceil(y) - y) * img1 + (y - math.floor(y)) * img3
                img6 = (math.ceil(y) - y) * img2 + (y - math.floor(y)) * img4

                img7 = (math.ceil(x) - x) * img5 + (x - math.floor(x)) * img6

                if (img7 - [0, 0, 0]).any() == False:
                    weight = 0.0

                acc[j, i, 0:3] += img7* weight

                acc[j, i, 3] += weight
    '''
    return acc
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
         输入:
         acc:输入图像，其alpha通道(第4通道)包含
         正常化的重量值
          输出:
          img: acc的r、g、b值归一化后的图像
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    #raise Exception("TODO in blend.py not implemented")
    '''h = acc.shape[0]
w = acc.shape[1]

img = acc[:, :, 0:3]/(acc[:, :, 3].reshape(h, w, 1))
'''
    h = acc.shape[0]
    w = acc.shape[1]
    img = np.zeros([h,w,3])
    
    for i in range(h):
        for j in range(w):
            if acc[i,j,3] > 0 :
                img[i,j] = acc[i,j,0:3]/acc[i,j,3]
            else :
                img[i,j] =[0,0,0]

    img = np.uint8(img)
    #TODO-BLOCK-END
    # END TODO

    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.
       这个函数获取由图像和相应转换组成的ImageInfo对象列表，并返回关于累积图像的有用信息

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
         由image (ImageInfo.img)和transform(image (ImageInfo.position)组成的ImageInfo对象列表。
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin

         accWidth:累加器图像的宽度(使所有已转换图像位于acc内的最小宽度)
         acch8:累加器图像高度(使所有已变形图像位于acc内的最小高度)
         通道:累加器映像中的通道数
         宽度:每个图像的宽度(假设:所有输入图像的宽度相同)
         平移:变换矩阵，使累加器图像左上角为原点
    """

    # Compute bounding box for the mosaic计算马赛克的边框框
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        # raise Exception("TODO in blend.py not implemented")
        x1, y1, x2, y2 = imageBoundingBox(img, M)
        minX = min(x1, minX)
        minY = min(y1, minY)
        maxX = max(x2, maxX)
        maxY = max(y2, maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
         输入:
         输入图象及其在镶嵌图中的相对位置一览表
         混合宽度:混合函数的宽度
         输出:
         croppedImage:通过混合所有图像和校正任何垂直漂移创建的最终马赛克
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    #raise Exception("TODO in blend.py not implemented")
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, outputWidth)

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

'''im=np.zeros([2,2,4])

im.fill(255)
im[0,0]=[10,20,30,40]
h = im.shape[0]
w = im.shape[1]
mg = np.zeros([h,w,3])
sd=im[:,:,3].reshape(2,2,1)
mg=im/sd

for i in range(h):
    for j in range(w):
        mg[i,j]= im[i,j,0:3]/im[i,j,3]

print(sd)
print(mg)'''