
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
        odd, compute the cross correlation of the given image with the given
        kernel, such that the output is of the same dimensions as the image and that
        you assume the pixels out of the bounds of the image to be zero. Note that
        you need to apply the kernel to each channel separately, if the given image
        is an RGB image.

        Inputs:
            img:    Either an RGB image (height x width x 3) or a grayscale image
                    (height x width) as a numpy array.
            kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                    equal).

        Output:
            Return an image of the same dimensions as the input image (same width,
            height and the number of color channels)'''
    m, n = kernel.shape
    m1 = (m - 1) // 2
    n1 = (n - 1) // 2
    c = img.ndim
    if c == 2:
        c = 1
    a = img.shape[0]
    b = img.shape[1]
    if c == 3:
        f = np.zeros([(a + m - 1), (b + n - 1), c], dtype=img.dtype)
        g = np.zeros([a, b, c], dtype=img.dtype)
    elif c == 1:
        f = np.zeros([(a + m - 1), (b + n - 1)], dtype=img.dtype)
        g = np.zeros([a, b], dtype=img.dtype)
    f[m1:a + m1, n1:b + n1] = img
    h = np.ravel(kernel)
    for i in range(a):
        for j in range(b):
            g[i, j] = np.dot(h, f[i:i + m, j:j + n].reshape(m * n, c))
    return g
def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    #对卷积核进行180度旋转
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    img = cross_correlation_2d(img,kernel)
    return img
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kernel = np.zeros([height, width])
    sum = 0
    for i in range(height):
        for j in range(width):
            kernel[i][j] = np.exp(-((height // 2-i)**2+(width // 2-j)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
            sum += kernel[i][j]
    kernel = kernel / sum
    return kernel
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    img = convolve_2d(img,kernel)
    return img
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    #img = img - low_pass(img, sigma, size)
    img = cv2.subtract(img, low_pass(img, sigma, size))
    return img
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    #high_low1 = high_low1.lower()
    #high_low2 = high_low2.lower()


    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

'''img1 = cv2.imread('cat(left).jpg')
img2 = cv2.imread('dog(right).jpg')

img1h = high_pass(img1, 8.6, 21)
img2l = low_pass(img2, 9.1, 19)
imghy = create_hybrid_image(img1, img2, 8.6, 21, 'high', 9.1, 19, 'low', 0.55)
#cv2.imshow("h", img1h)
#cv2.imshow("l", img2l)
cv2.imshow("hy", imghy)
#cv2.imwrite('img1hfalse.jpg', img1h)
#cv2.imwrite('img2l.jpg', img2l)
#cv2.imwrite('hybrid.jpg', imghy)
cv2.waitKey(0)
cv2.destroyAllWindows()
,,,