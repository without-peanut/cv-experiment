# -*- coding: utf-8 -*-
import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    输入:
        f1——cv2的列表。第一个图像中的关键点对象
        f2——cv2的列表。第二幅图像中的关键点对象
        matches——cv2的列表。DMatch对象
        DMatch.queryIdx:第一张图像中特征的索引
        DMatch.trainIdx:第二幅图像中特征的索引
        DMatch.distance:两个特征之间的距离
        A_out——忽略这个参数。如果其他todo需要computeHomography，调用computeHomography(f1,f2,matches)
    输出:H——二维单应性(3x3矩阵)
    取特征的两个列表f1和f2，以及特征匹配的一个列表，并从匹配的图像1到图像2估计一个单应性。
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0齐次线性方程中A矩阵的维数Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #在这个循环中填入矩阵A。使用方括号访问元素。例如A[0,0]
        #TODO-BLOCK-BEGIN
        
        A[2 * i, 0] = a_x
        A[2 * i, 1] = a_y
        A[2 * i, 2] = 1
        A[2 * i, 3:6] = 0
        A[2 * i, 6] = -b_x * a_x
        A[2 * i, 7] = -b_x * a_y
        A[2 * i, 8] = -b_x

        A[2 * i + 1, 0:3] = 0
        A[2 * i + 1, 3] = a_x
        A[2 * i + 1, 4] = a_y
        A[2 * i + 1, 5] = 1
        A[2 * i + 1, 6] = -b_y * a_x
        A[2 * i + 1, 7] = -b_y * a_y
        A[2 * i + 1, 8] = -b_y
                       
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order s是按降序排列的一维奇异值数组
    #U, Vt are unitary matrices U, Vt是正交矩阵
    #Rows of Vt are the eigenvectors of A^TA.Vt的行向量是A ^ TA的特征向量
    #Columns of U are the eigenvectors of AA^T.U的列向量是AA ^ T的特征向量。

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD用SVD的适当元素填充单应性H
    #TODO-BLOCK-BEGIN
    
    H = Vt[-1].reshape((3,3))
    
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    输入:
        f1——cv2的列表。第一个图像中的关键点对象
        f2——cv2的列表。第二幅图像中的关键点对象
        matches——cv2的列表。DMatch对象
            DMatch.queryIdx:第一张图像中特征的索引
            DMatch.trainIdx:第二幅图像中特征的索引
            DMatch.distance:两个特征之间的距离
        m——运动模型(eTranslate, eHomography)
        nRANSAC——RANSAC迭代次数
        RANSACthresh——RANSAC距离阈值
    输出:
        M——图像间变换矩阵
        nRANSAC迭代重复:
            选择最小的特性匹配集。
            估计这些匹配所隐含的转换计算离群值的数量。
        对于最大离群值个数的变换，
        计算最小二乘运动估计使用的离群值，
        然后以变换矩阵M的形式返回。
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    '''写出整个方法。您需要处理两种类型的运动模型，
    纯平移(m == eTranslation)和完全同质(m == eHomography)。
    然而，您应该只有一个外部循环来执行RANSAC代码，
    因为这两种情况下RANSAC的使用几乎是相同的。
    
    您的单应性处理代码应该调用compute_homography。
    这个函数还应该调用get_inliers，最后调用least_squares_fit。'''
    #TODO-BLOCK-BEGIN
    
    max_i = []
    for i in range(nRANSAC):
        
        if m == eHomography:           
            matchesH = np.random.choice(matches, 4)#random几种函数的选用
            H = computeHomography(f1, f2, matchesH)

        elif m == eTranslate:
            matchesT = np.random.choice(matches)
            H = np.eye(3)
            H[0,2] = f2[matchesT.trainIdx].pt[0] - f1[matchesT.queryIdx].pt[0]
            H[1,2] = f2[matchesT.trainIdx].pt[1] - f1[matchesT.queryIdx].pt[1]
        
        else:
            raise Exception("Error: Invalid motion model.")

        inlier_indices = getInliers(f1, f2, matches, H, RANSACthresh)

        if len(inlier_indices) > len(max_i): 
            max_i = inlier_indices

    M = leastSquaresFit(f1, f2, matches, m, max_i)
    
    
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    输入:
        f1——cv2的列表。第一个图像中的关键点对象
        f2——cv2的列表。第二幅图像中的关键点对象
        matches——cv2的列表。DMatch对象
            DMatch.queryIdx:第一张图像中特征的索引
            DMatch.trainIdx:第二幅图像中特征的索引
            DMatch.距离:两个特征之间的距离
        M——图像间变换矩阵
        RANSACthresh——RANSAC距离阈值
    输出:
        inlier_indexes——inlier匹配指数("matches"里的索引)
        
        对f1中的匹配特征进行M变换。
        将变换后的特征在f1中的匹配索引存储在f2中匹配特征的欧氏距离搜索范围内。
        返回这些特性的匹配索引数组。
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #判断第i个匹配特征f1[id1]被M变换后是否在f2匹配特征的RANSACthresh内。
        #如果是，将i附加到inliers
        #TODO-BLOCK-BEGIN
        
        (x1, y1) = f1[matches[i].queryIdx].pt
        (x2, y2) = f2[matches[i].trainIdx].pt
        
        a = np.zeros(3)

        a[:] = [x1, y1, 1]
        b = np.dot(M, a)
        
        if b[-1] > 10 ** -5:
            x3, y3, z3 = b / b[-1]
            dist = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

            if dist <= RANSACthresh:
                inlier_indices.append(i)
        
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    输入:
        f1——cv2的列表。第一个图像中的关键点对象
        f2——cv2的列表。第二幅图像中的关键点对象
        matches——cv2的列表。DMatch对象
            DMatch.queryIdx:第一张图像中特征的索引
            DMatch.trainIdx:第二幅图像中特征的索引
            DMatch.distance:两个特征之间的距离
        m——运动模型(eTranslate, eHomography)
        inlier_indexes——inlier匹配索引(进入“matches”的索引)
    输出:
        M -变换矩阵
        只使用内点计算从f1到f2的变换矩阵并返回它。
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).
    #这个函数需要处理两种可能的运动模型，纯平移(eTranslate)和完全同质(eHomography)。

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.
        '''对于球面扭曲的图像，转换是平移，只有两个自由度。
        因此，对于所有的内点，我们只需计算f1中的特征与其在f2中的匹配之间的平均平移向量'''

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.使用此循环计算所有内点的平均平移向量。
            #TODO-BLOCK-BEGIN
            j = inlier_indices[i]
            u += f2[matches[j].trainIdx].pt[0] - f1[matches[j].queryIdx].pt[0]
            v += f2[matches[j].trainIdx].pt[1] - f1[matches[j].queryIdx].pt[1]
                                                           
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        
        i_matches = []
        
        for i in inlier_indices:
            i_matches.append(matches[i])
            
        M = computeHomography(f1, f2, i_matches)
                       
                                                               
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

