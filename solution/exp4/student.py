# -*- coding: utf-8 -*-
# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.
    给定从同一视点拍摄的一组图像和相应的光源方向，该函数计算朗伯场景的反照率和法线图。
    如果一个像素的计算反照率的L2范数小于1e-7，那么将反照率设为黑色，并将法线设为0向量。
    法线应该是单位向量。

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    输入:
    灯光——nx3阵列。行被规范化，并被解释为照明方向。
    图像——N个图像的列表。每幅图像都是从相同的视点拍摄的同一场景，但是在灯光中指定的光照条件下。
    输出:
    反照率——N浮动32高×宽×3图像与尺寸匹配的输入图像。
    法线——浮动32高x宽x 3图像与尺寸匹配的输入图像。
    """
    #raise NotImplementedError()
    #n = lights.shape[0]  #图片个数

    height, width, channel = images[0].shape
    albedo = np.zeros((height, width, channel), dtype = np.float32)
    normals = np.zeros((height, width, 3), dtype = np.float32)
    #intensity = np.zeros(n, 3, dtype = np.uint8)
    arrimages = np.array(images) #将图片列表转为多维数组n*h*w*3
    lights = np.matrix(lights)
    for i in range(height):
        for j in range(width):
            intensity = arrimages[:, i, j, :]
            intensity = np.matrix(intensity).astype(np.float32)
            G3 = (((lights.T).dot(lights)).I).dot((lights.T).dot(intensity))
            if channel == 3:
                albedo[i, j, 0] = np.linalg.norm(G3[:, 0])
                albedo[i, j, 1] = np.linalg.norm(G3[:, 1])
                albedo[i, j, 2] = np.linalg.norm(G3[:, 2])
                if albedo[i, j, 0] < 1e-7:
                    albedo[i, j, :] = 0
                    normals[i, j, :] = [0, 0, 0]
                else:
                    normals[i, j, :] = (G3[:, 0] / albedo[i, j, 0]).reshape(-1)
            elif channel == 1:
                albedo[i, j] = np.linalg.norm(G3)
                if albedo[i, j] < 1e-7:
                    albedo[i, j] = 0
                    normals[i, j, :] = [0, 0, 0]
                else:
                    normals[i, j, :] = (G3/ albedo[i, j]).reshape(-1)

    return albedo, normals
def project_impl(K, Rt, points):
    """
         Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    投影三维点到一个校准的相机。
    输入:
        K——摄像机内部标定矩阵
        Rt—3×4摄像机外标定矩阵
        点——高度x宽度x 3个三维点数组
    输出:
        投影——2D投影阵列的高度x宽度x 2

    """
    height = points.shape[0]
    width = points.shape[1]
    projections = np.zeros((height, width, 2))

    M = K.dot(Rt)

    for i in range(height):
        for j in range(width):

            p = np.append(points[i, j], 1)
            p = M.dot(p)
            projections[i, j, 0] = p[0] / p[2]
            projections[i, j, 1] = p[1] / p[2]

    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
        Prepare normalized patch vectors according to normalized cross
        correlation.

        This is a preprocessing step for the NCC pipeline.  It is expected that
        'preprocess_ncc' is called on every input image to preprocess the NCC
        vectors and then 'compute_ncc' is called to compute the dot product
        between these vectors in two images.

        NCC preprocessing has two steps.
        (1) Compute and subtract the mean.
        (2) Normalize the vector.

        The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
        patch, compute the R, G, and B means separately.  The normalization
        is over all channels.  i.e. For an RGB image, after subtracting out the
        RGB mean, compute the norm over the entire (ncc_size**2 * channels)
        vector and divide.

        If the norm of the vector is < 1e-6, then set the entire vector for that
        patch to zero.

        Patches that extend past the boundary of the input image at all should be
        considered zero.  Their entire vector should be set to 0.

        Patches are to be flattened into vectors with the default numpy row
        major order.  For example, given the following
        2 (height) x 2 (width) x 2 (channels) patch, here is how the output
        vector should be arranged.

        channel1         channel2
        +------+------+  +------+------+ height
        | x111 | x121 |  | x112 | x122 |  |
        +------+------+  +------+------+  |
        | x211 | x221 |  | x212 | x222 |  |
        +------+------+  +------+------+  v
        width ------->

        v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

        see order argument in np.reshape

        Input:
            image -- height x width x channels image of type float32
            ncc_size -- integer width and height of NCC patch region.
        Output:
            normalized -- height x width x (channels * ncc_size**2) array

        根据归一化相关关系，编制归一化patch向量。
        这是NCC管道的预处理步骤。我们期望在每个输入图像上调用preprocess_ncc
        来预处理NCC向量，然后调用compute_ncc来计算两个图像中这些向量之间的点积。
        NCC预处理有两个步骤。
        (1)计算并减去平均值。
        (2)对向量进行归一化。
        平均值是每个通道。对于RGB图像，在ncc_size**2 patch上分别计算R、G和B。
        标准化是在所有通道上进行的。也就是说，对于一个RGB图像，在减去RGB平均值之后，
        计算整个(ncc_size**2 * channels)向量的范数，然后除以。
        如果向量的范数小于1e-6，那么将这个patch的整个向量设为零。
        超过输入图像边界的补片应视为零。它们的整个向量应该被设为0。
        补丁将以默认的numpy行主顺序被压扁为向量。例如，给定下列条件
        2(高)x 2(宽)x 2(通道)贴片，这里是输出向量的排列方式。
            channel1         channel2
        +------+------+  +------+------+ height
        | x111 | x121 |  | x112 | x122 |  |
        +------+------+  +------+------+  |
        | x211 | x221 |  | x212 | x222 |  |
        +------+------+  +------+------+  v
        width ------->

        v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

        参见npreshape中的order参数

        输入:
            image——高度x宽度x通道类型为浮点32的图像
            ncc_size——NCC patch区域的整数宽度和高度。
        输出:
            normalized——height x width x (channels * ncc_size**2)数组

    """

    height, width, channels = image.shape
    normalized = np.zeros((height, width, channels * ncc_size ** 2))

    a = ncc_size // 2

    for i in range(a, height - a):
        for j in range(a, width - a):
            v = []

            for c in range(channels):
                patch = image[i - a: i + a + 1, j - a: j + a + 1, c]
                average = np.mean(patch)
                v = np.append(v, [(patch - average).flatten()])

            L2 = np.linalg.norm(v)

            if L2 < 1e-6:
                normalized[i, j] = np.zeros((v.shape))
            else:
                normalized[i, j] = v / L2

    return normalized



def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    计算两个已经用preprocess_ncc计算每个像素的标准化向量的图像之间的归一化相互关系。
    输入:
        image1——高x宽x (channels * ncc_size**2)数组
        image2——高x宽x (channels * ncc_size**2)数组
    输出:
        ncc——图像1与图像2的高度x宽度归一化互相关。
    """
    height, width, channels = image1.shape

    ncc1 = np.multiply(image1.reshape(height*width, -1), image2.reshape(height*width, -1))
    ncc2 = np.sum(ncc1, axis = 1)
    ncc = ncc2.reshape(height,width)

    return ncc



def form_poisson_equation_impl(height, width, alpha, normals, depth_weight, depth):
    """
    Creates a Poisson equation given the normals and depth at every pixel in image.
    The solution to Poisson equation is the estimated depth. 
    When the mode, is 'depth' in 'combine.py', the equation should return the actual depth.
    When it is 'normals', the equation should integrate the normals to estimate depth.
    When it is 'both', the equation should weight the contribution from normals and actual depth,
    using  parameter 'depth_weight'.

    Input:
        height -- height of input depth,normal array
        width -- width of input depth,normal array
        alpha -- stores alpha value of at each pixel of image. 
            If alpha = 0, then the pixel normal/depth should not be 
            taken into consideration for depth estimation
        normals -- stores the normals(nx,ny,nz) at each pixel of image
            None if mode is 'depth' in combine.py
        depth_weight -- parameter to tradeoff between normals and depth when estimation mode is 'both'
            High weight to normals mean low depth_weight.
            Giving high weightage to normals will result in smoother surface, but surface may be very different from
            what the input depthmap shows.
        depth -- stores the depth at each pixel of image
            None if mode is 'normals' in combine.py
    Output:
        constants for equation of type Ax = b
        A -- left-hand side coefficient of the Poisson equation 
            note that A can be a very large but sparse matrix so csr_matrix is used to represent it.
        b -- right-hand side constant of the the Poisson equation
    """

    assert alpha.shape == (height, width)
    assert normals is None or normals.shape == (height, width, 3)
    assert depth is None or depth.shape == (height, width)

    '''
    Since A matrix is sparse, instead of filling matrix, we assign values to a non-zero elements only.
    For each non-zero element in matrix A, if A[i,j] = v, there should be some index k such that, 
        row_ind[k] = i
        col_ind[k] = j
        data_arr[k] = v
    Fill these values accordingly
    '''
    row_ind = []
    col_ind = []
    data_arr = []
    '''
    For each row in the system of equation fill the appropriate value for vector b in that row
    '''
    b = []
    if depth_weight is None:
        depth_weight = 1

    '''
    TODO
    Create a system of linear equation to estimate depth using normals and crude depth Ax = b

    x is a vector of depths at each pixel in the image and will have shape (height*width)

    If mode is 'depth':
        > Each row in A and b corresponds to an equation at a single pixel
        > For each pixel k, 
            if pixel k has alpha value zero do not add any new equation.
            else, fill row in b with depth_weight*depth[k] and fill column k of the corresponding
                row in A with depth_weight.

        Justification: 
            Since all the elements except k in a row is zero, this reduces to 
                depth_weight*x[k] = depth_weight*depth[k]
            you may see that, solving this will give x with values exactly same as the depths, 
            at pixels where alpha is non-zero, then why do we need 'depth_weight' in A and b?
            The answer to this will become clear when this will be reused in 'both' mode

    Note: The normals in image are +ve when they are along an +x,+y,-z axes, if seen from camera's viewpoint.
    If mode is 'normals':
        > Each row in A and b corresponds to an equation of relationship between adjacent pixels
        > For each pixel k and its immideate neighbour along x-axis l
            if any of the pixel k or pixel l has alpha value zero do not add any new equation.
            else, fill row in b with nx[k] (nx is x-component of normal), fill column k of the corresponding
                row in A with -nz[k] and column k+1 with value nz[k]
        > Repeat the above along the y-axis as well, except nx[k] should be -ny[k].

        Justification: Assuming the depth to be smooth and almost planar within one pixel width.
        The normal projected in xz-plane at pixel k is perpendicular to tangent of surface in xz-plane.
        In other word if n = (nx,ny,-nz), its projection in xz-plane is (nx,nz) and if tangent t = (tx,0,tz),
            then n.t = 0, therefore nx/-nz = -tz/tx
        Therefore the depth change with change of one pixel width along x axis should be proporational to tz/tx = -nx/nz
        In other words (depth[k+1]-depth[k])*nz[k] = nx[k]
        This is exactly what the equation above represents.
        The negative sign in ny[k] is because the indexing along the y-axis is opposite of +y direction.

    If mode is 'both':
        > Do both of the above steps.

        Justification: The depth will provide a crude estimate of the actual depth. The normals do the smoothing of depth map
        This is why 'depth_weight' was used above in 'depth' mode. 
            If the 'depth_weight' is very large, we are going to give preference to input depth map.
            If the 'depth_weight' is close to zero, we are going to give preference normals.
    '''
    #TODO Block Begin
    #fill row_ind,col_ind,data_arr,b
    row = 0
    if depth is not None:
        for i in range(height):
            for j in range(width):
                if alpha[i, j] != 0:
                    row_ind.append(row)
                    row += 1
                    col_ind.append(i*width+j)
                    data_arr.append(depth_weight)
                    b.append(depth_weight*depth[i, j])
    if normals is not None:
        for i in range(height):
            for j in range(width):
                if alpha[i, j] != 0 and \
                        (i+1 < height and j+1 < width) and \
                        (alpha[i+1, j] != 0 and alpha[i, j+1] != 0):
                    row_ind.append(row)
                    col_ind.append(i*width+j)
                    data_arr.append(-normals[i, j, 2])
                    row_ind.append(row)
                    col_ind.append(i*width+j+1)
                    data_arr.append(normals[i, j, 2])
                    row += 1
                    b.append(normals[i, j, 0])
                    row_ind.append(row)
                    col_ind.append(i*width+j)
                    data_arr.append(-normals[i, j, 2])
                    row_ind.append(row)
                    col_ind.append((i+1)*width+j)
                    data_arr.append(normals[i, j, 2])
                    row += 1
                    b.append(-normals[i, j, 1])
    #TODO Block end
    # Convert all the lists to numpy array
    row_ind = np.array(row_ind, dtype=np.int32)
    col_ind = np.array(col_ind, dtype=np.int32)
    data_arr = np.array(data_arr, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Create a compressed sparse matrix from indices and values
    A = csr_matrix((data_arr, (row_ind, col_ind)), shape=(row, width * height))

    return A, b



