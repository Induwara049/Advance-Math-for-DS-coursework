import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.fftpack as sfft
import scipy.signal as signal

image = mpimg.imread(r"./Fruit.jpg")

## part (a)  ##

# fft
imgfft = sfft.fft2(image)
plt.imshow(np.abs(imgfft))
plt.show()

# image with fft shift
imgf = sfft.fftshift(imgfft)
plt.imshow(np.abs(imgf))
plt.show()

# removing low frequencies
imgf1 = np.zeros((360, 360), dtype=complex)
c = 180
r = 90
for m in range(0, 360):
    for n in range(0, 360):
        if np.sqrt(((m - c) ** 2 + (n - c) ** 2)) > r:
            imgf1[m, n] = imgf[m, n]

image1 = sfft.ifft2(imgf1)
plt.imshow(np.abs(image1))
plt.show()

## part (b)  ##

# Gaussian filter
kernel = np.outer(signal.gaussian(360, 5), signal.gaussian(360, 5))
kf = sfft.fft2(sfft.ifftshift(kernel))  # freq domain kernel
plt.imshow(np.abs(kf))

# plt.show()
imgf = sfft.fft2(image)
plt.imshow(np.abs(kf))

# plt.show()
img_b = imgf * kf
plt.imshow(np.abs(img_b))

# plt.show()
image1 = sfft.ifft2(img_b)
plt.imshow(np.abs(image1))
plt.show()

##  part (c)  ##

# #DCT
imgc = sfft.dct((sfft.dct(image, norm='ortho')).T, norm='ortho')
plt.imshow(imgc)
# plt.show()

# IDCT
image1 = sfft.idct((sfft.idct(imgc, norm='ortho')).T, norm='ortho')
plt.imshow(image1)
plt.show()

# Scaling
imgc2 = imgc[0:240, 0:240]
image1 = sfft.idct((sfft.idct(imgc2, norm='ortho')).T, norm='ortho')
plt.imshow(image1)
plt.show()

## part (d)  ##

# Removing high frequency components
imgc1 = np.zeros((480, 480))
imgc1[:120, :120] = imgc[:120, :120]
image1 = sfft.idct((sfft.idct(imgc1, norm='ortho')).T, norm='ortho')
plt.imshow(image1)
plt.show()
