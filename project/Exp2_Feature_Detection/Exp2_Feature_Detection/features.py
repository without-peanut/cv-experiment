# coding:utf-8
import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        输入:
            图像——uint8 BGR图像，值在[0,255]之间
        输出:
            检测到的关键点列表，填写cv2。关键点对象用被检测关键点的坐标，梯度的角度
            (以度为单位)，检测器响应(Harris评分为Harris检测器)，设置大小为10。
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    计算机愚蠢的示例特性。这并没有做任何有意义的事情，但是作为一个例子可能有用。
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        保存harrisImage的可视化,通过覆盖Harris响应srcImage红色的形象。
        输入:
            srcImage——带有[0,1]值的numpy数组中的灰度输入图像。维度是(行，列)。
            harrisImage——numpy数组中的灰度输入图像，值在[0,1]中。维度是(行，列)。
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        #制作一个灰度背景
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        输入:
            srcImage——带有[0,1]值的numpy数组中的灰度输入图像。维度是(行，cols)。
        输出:
            harrisImage——numpy数组，每个像素包含Harris分数。
            orientationImage——numpy数组，包含每个像素的梯度方向(以角度计)。
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        '''
        计算“srcImage”的哈里斯角强度，并存储在“harrisImage”中。
        关于如何做到这一点，请参阅项目页面。还要计算每个像素的方向，
        并将其存储在“orientationImage”中。
        '''
        # TODO-BLOCK-BEGIN
        
        Ix = scipy.ndimage.sobel(srcImage, 1)
        Iy = scipy.ndimage.sobel(srcImage, 0)

        orientationImage = np.degrees(np.arctan2(Iy, Ix))
        
        A = scipy.ndimage.gaussian_filter(Ix * Ix, sigma=0.5)
        B = scipy.ndimage.gaussian_filter(Ix * Iy, sigma=0.5)
        C = scipy.ndimage.gaussian_filter(Iy * Iy, sigma=0.5)
        
        det_H = A * C - B ** 2
        trace_H = A + C
        
        harrisImage = det_H - 0.1 * trace_H ** 2
        
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        输入：
            harrisImage——numpy数组，每个像素包含Harris分数。
        输出:
            destImage——在每个像素处包含真/假的numpy数组，取
                      决于像素值是否为其7x7邻域的局部最大值
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        destImage = (harrisImage == scipy.ndimage.maximum_filter(harrisImage, size=7))
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        输入:
            图像——BGR图像值在[0,255]之间
        输出:
            检测到的关键点列表，填写cv2。关键点对象用被检测关键点的坐标，梯度的角
            度(以度为单位)，检测器响应(Harris评分为Harris检测器)，设置大小为10。
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        #创建用于哈里斯检测的灰度图像
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        '''
        computeHarrisValues()计算每个像素处的harris分数将结果存储在harrisImage中。
        您需要实现这个函数。
        '''
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        '''
        计算哈里斯图像中的局部最大值。您需要实现这个函数。
        创建图像以存储局部最大哈里斯值为真，其他像素为假
        '''
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        '''循环遍历harrisMaxImage中的特征点，
        并为每个点填写描述符计算所需的信息。你需要填充x y和角度。
        '''
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score               
                # TODO-BLOCK-BEGIN
                f.pt = (x, y)
                f.size = 10
                f.angle = orientationImage[y,x]
                f.response = harrisImage[y,x]
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        输入:
            图像——BGR图像值在[0,255]之间
            关键点——检测到的特征，我们必须在指定的坐标上计算特征描述符
        输出:desc——kx25 numpy数组，其中K是关键点的个数
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))
        
        m,n = grayImage.shape
        newimage = np.zeros([m+4, n+4])
        newimage[2:m + 2, 2:n + 2] = grayImage

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            '''简单描述符是一个以特征点为中心采样的5x5强度窗口。
               将描述符存储为行主向量。将图像外的像素视为零。'''
            # TODO-BLOCK-BEGIN
            
            desc[i, :] = newimage[y:y+5, x:x+5].reshape(1,-1)           

            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        输入:
            图像——BGR图像值在[0,255]之间
            关键点——检测到的特征，我们必须在指定的坐标上计算特征描述符
        输出:
            desc——K x W^2 的numpy数组,K是Keykpoint的数量,W是窗口大小
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        #此图像表示需要计算以存储为特征描述符(行-主)的特性周围的窗口
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            '''根据特征位置/方向来计算转换。您需要计算从围绕该特征的40x40旋转窗口
               中的每个像素到8x8特征描述符图像中的适当像素的转换。'''
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
                                   
            angle = f.angle
            angle = (angle / 180)*np.pi
            x, y = f.pt
            T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
            R = np.array([[np.cos(angle), np.sin(angle), 0],
                          [-np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
            S = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])
            T2 = np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]])
            transMx = np.dot(np.dot(np.dot(T2, S), R), T1)[0:2, 0:3]

            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            #调用warpaffine函数进行映射，它需要一个2x3矩阵
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            '''将描述符规范化为零均值和单位方差。
               如果方差为0，则将描述符向量设置为0。最后，将向量写到desc。'''
            # TODO-BLOCK-BEGIN
            
            window = destImage[:windowSize, :windowSize]
            if np.std(window) <= (10**-5):
                desc[i, :] = np.zeros([1,windowSize * windowSize])
            else:
                desc[i, :] = ((window - np.mean(window))/np.std(window)).reshape(1,-1)
                       
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        输入:
            图像——BGR图像值在[0,255]之间
            关键点——检测到的特征，我们必须在指定的坐标上计算特征描述符
        输出:
            描述符numpy数组，尺寸:
                关键点数x特征描述符维度
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        输入:
            desc1——存储在numpy数组中的图像1的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
            desc2——存储在numpy数组中的图像2的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
        输出:
            特性匹配:一个cv2列表。DMatch对象
                如何设置属性:
                    queryIdx:第一张图像中特征的索引
                    trainIdx:第二个图像中特征的索引
                    距离:两个特征之间的距离
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    #使用地面真值单应性评估匹配。计算匹配特征点与实际变换位置之间的平均SSD距离。
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        输入:
            desc1——存储在numpy数组中的图像1的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
            desc2——存储在numpy数组中的图像2的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
        输出:
            特性匹配:一个cv2列表。DMatch对象
                如何设置属性:
                    queryIdx:第一张图像中特征的索引
                    trainIdx:第二个图像中特征的索引
                    距离:两个特征之间的距离
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        '''执行简单的功能匹配。
        该方法使用两个特征向量之间的SSD距离，
        并将第一幅图像中的特征与第二幅图像中最接近的特征匹配。
        注意:第一幅图像中的多个特征可能与第二幅图像中的相同特征相匹配。'''
        # TODO-BLOCK-BEGIN
        

        dist = scipy.spatial.distance.cdist(desc1, desc2)
        for i in range(desc1.shape[0]):
            j = np.argmin(dist[i])
            m = cv2.DMatch()
            m.queryIdx = i
            m.trainIdx = j
            m.distance = dist[i][j]
            matches.append(m)
            
            
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        输入:
            desc1——存储在numpy数组中的图像1的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
            desc2——存储在numpy数组中的图像2的特征描述符，
            维度:行(关键点数)x列(特征描述符的维度)
        输出:
            特性匹配:一个cv2列表。DMatch对象
                如何设置属性:
                    queryIdx:第一张图像中特征的索引
                    trainIdx:第二个图像中特征的索引
                    距离:测试分数的比值
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        '''执行比值特征匹配。它使用两个最佳匹配的SSD距离的比值，
           并将第一幅图像中的一个特征与第二幅图像中最接近的特征匹配。
           注意:第一幅图像中的多个特征可能与第二幅图像中的相同特征相匹配。
           在这个函数中不需要阈值匹配'''
        # TODO-BLOCK-BEGIN

        dist = scipy.spatial.distance.cdist(desc1, desc2)
        for i in range(desc1.shape[0]):
            j = np.argmin(dist[i])
            dist1 = dist.copy()
            dist1[i][j] = float('inf')
            k = np.argmin(dist1[i])
            m = cv2.DMatch()
            m.queryIdx = i
            m.trainIdx = j
            m.distance = dist[i][j] / dist[i][k]
            matches.append(m)
            
            
            
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

