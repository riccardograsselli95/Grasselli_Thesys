"""
##############################################################################
##############################################################################
########                                                       ###############
########                   RICCARDO GRASSELLI                  ###############
########                                                       ###############
##############################################################################
##############################################################################


"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# if example == True genererà una sola immagine
# altrimenti sottocampionerà tutto il dataset
example = True
percentage_to_cover = 45 # indica la percentuale di sottocampionamento dell'immagine
dataset_path = 'dataset/'
subsampled_path = str(percentage_to_cover)+'/0/'
name_newsubsampleimage = 'input'


# come deve essere il valore di circle_ray in base alla percentuale che vogliamo ottenere:
# 45% -> 90
# 35% -> 80
# 25% -> 70
# 17% -> 55
# 11% -> 45
if percentage_to_cover >= 45:
    circle_ray = 90
elif percentage_to_cover >= 35:
    circle_ray = 80
elif percentage_to_cover >= 25:
    circle_ray = 70
elif percentage_to_cover >= 15:
    circle_ray = 55
elif percentage_to_cover >= 10:
    circle_ray = 45
elif percentage_to_cover >= 5:
    circle_ray = 25
elif percentage_to_cover < 5:
    circle_ray = 15
dataset_listimages = os.listdir(dataset_path)
try:
    os.mkdir(subsampled_path)
except OSError:
    print ("Creation of the directory %s failed" % subsampled_path)
else:
    print ("Successfully created the directory %s " % subsampled_path)

def subsample_image(dir1, dir2, image_path, save_path, percent, new_name, visualize):
    number = image_path[6:]
    print('*************************')
    print('IMAGE NAME: ', number)
    img = cv2.imread(dir1+image_path, 0) # load an image
    #print('IMAGE SHAPE: ', img.shape, len(img[0]), ' - ', img[0])

    #Output is a 2D complex array. 1st channel real and 2nd imaginary
    #For fft in opencv input image needs to be converted to float32
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    #Rearranges a Fourier transform X by shifting the zero-frequency 
    #component to the center of the array.
    #Otherwise it starts at the tope left corenr of the image (array)
    dft_shift = np.fft.fftshift(dft)
    #print('DFT: ', dft_shift.shape, len(dft_shift[0][0]))

    ##Magnitude of the function is 20.log(abs(f))
    #For values that are 0 we may end up with indeterminate values for log. 
    #So we can add 1 to the array to avoid seeing a warning. 
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


    # Circular HPF mask, center circle is 0, remaining all ones
    #Can be used for edge detection because low frequencies at center are blocked
    #and only high frequencies are allowed. Edges are high frequency components.
    #Amplifies noise.

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = percent
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    print(mask_area)
    mask[mask_area] = 1


    zeri = np.count_nonzero(mask==0)
    uni = np.count_nonzero(mask==1)
    total = img.shape[0] * img.shape[1]
    percentage = uni / (total*2)
    if visualize:
        print('REAL PERCENTAGE of SUBSAMPLING: ', percentage)

    # apply mask and inverse DFT: Multiply fourier transformed image (values)
    #with the mask values. 
    fshift = dft_shift * mask

    #Get the magnitude spectrum (only for plotting purposes)
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    #Inverse shift to shift origin back to top left.
    f_ishift = np.fft.ifftshift(fshift)

    #Inverse DFT to convert back to image domain from the frequency domain. 
    #Will be complex numbers
    img_back = cv2.idft(f_ishift)

    #Magnitude spectrum of the image domain
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # cmap = plt.cm.jet
    # norm = plt.Normalize(vmin=img_back.min(), vmax=img_back.max())
    # image = cmap(norm(img_back))
    # plt.imsave(dir2+save_path+number, image)
    plt.imsave(dir2+new_name+number, img_back, cmap='gray')

    if visualize:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('After inverse FFT')
        plt.show()


#for image_to_process in entries:
if example:
    for image_to_process in [dataset_listimages[0]]:
        subsample_image(dataset_path, subsampled_path, image_to_process, 'input', circle_ray, name_newsubsampleimage, True)
else:
    for image_to_process in dataset_listimages:
        print('image_to_process ', image_to_process)
        subsample_image(dataset_path, subsampled_path, image_to_process, 'input', circle_ray, name_newsubsampleimage, False)




