"""
##############################################################################
##############################################################################
########                                                       ###############
########                   RICCARDO GRASSELLI                  ###############
########                                                       ###############
##############################################################################
##############################################################################


"""

from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# https://stackoverflow.com/questions/50387606/python-draw-line-between-two-coordinates-in-a-matrix
def draw_line(mat, x0, y0, x1, y1, inplace=False):
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 2
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        #mat = mat.T
        mat = np.transpose(mat, (1, 0, 2))
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends
    mat[x0, y0] = 1
    mat[x1, y1] = 1
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[x, y] = 1
    if not inplace:
        if not transpose:
            return mat 
        else:
            mat = np.transpose(mat, (1, 0, 2))
            return mat




# if example == True genererà una sola immagine
# altrimenti sottocampionerà tutto il dataset
example = False
spokes = 2 # 2=8 4=16 8=32 12=48 or 16=64
dataset_path = 'dataset/'
subsampled_path = str(spokes*4)+'/0/'
name_newsubsampleimage = 'input'


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
    #print(img.shape)
    crow, ccol = int(rows / 2), int(cols / 2)
    # print('crow: ', crow)
    # print('ccol: ', ccol)
    numb_of_spoke = percent # 8=32 12=48 or 16=64
    space_x = int(ccol / numb_of_spoke)
    space_y = int(crow / numb_of_spoke)   
    # print('space_x: ', space_x)
    # print('space_y: ', space_y)
    points_x = np.arange(0, cols, space_x, dtype=int)
    points_y = np.arange(0, rows, space_y, dtype=int)
    if numb_of_spoke != 1:
        points_x[-1] = img.shape[1]-1
        points_y[-1] = img.shape[0]-1
    points_xflip = np.flip(points_x)
    points_y = np.flip(points_y)
    points_yflip = np.flip(points_y)
    # print('points_x: ', points_x)
    # print('points_y: ', points_y)
    # print('points_xflip: ', points_xflip)
    # print('points_yflip: ', points_yflip)


    #plt.plot([points_x[0],points_xflip[0]], [points_y[0],points_yflip[0]])
    
    mask = np.zeros((rows, cols, 2), np.uint8)
    #mask_area = np.zeros((rows, cols), np.uint8)
    #print(mask.size)
    # mask = draw_line(mask, points_x[0],points_y[0],points_xflip[0], points_yflip[0], inplace=False)
    # mask = draw_line(mask, points_xflip[0],points_y[0],points_x[0], points_yflip[0], inplace=False)
    #mask = draw_line(mask, points_xflip[8],points_y[16],points_xflip[8], points_y[0], inplace=False)
    #mask = draw_line(mask, 0,184, 367,184, inplace=False)
    #print(mask_area)
    width_1 = 1
    width_2 = width_1 - 1
    a = 0
    for j in points_x:
        #print('J: ', j, ' - ', rows)
        if j == 0:
            for i in range(j,j+width_1):
                #print('points 1: ', points_yflip[0], ' - ',int(points_xflip[0]-i) , ' - ',points_y[0], ' - ', i)
                mask = draw_line(mask, points_yflip[0],int(points_xflip[0]-i),points_y[0], i, inplace=False)
        elif j == (rows-width_2):
            for i in range(j-width_2,j):
                #print('points 1: ', points_yflip[0], ' - ',int(points_xflip[0]-i) , ' - ',points_y[0], ' - ', i)
                mask = draw_line(mask, points_yflip[0],int(points_xflip[0]-i),points_y[0], i, inplace=False)
        else:
            for i in range(j-width_2,j+width_1):
                if not(int(points_xflip[0]-i) < 0):
                    #print('points 1: ', points_yflip[0], ' - ',int(points_xflip[0]-i) , ' - ',points_y[0], ' - ', i)
                    mask = draw_line(mask, points_yflip[0],int(points_xflip[0]-i),points_y[0], i, inplace=False)

    # print('*****************')
    # print('points_x: ', points_x)
    # print('points_y: ', points_y)
    # print('points_xflip: ', points_xflip)
    # print('points_yflip: ', points_yflip)

    for j in points_y:
        #print('J: ', j, ' - ', cols)
        if j == 0:
            for i in range(j,j+width_1):
                #print('POINTS1: ', int(points_y[0]-i), ' - ', points_xflip[0], ' - ',i, ' - ', points_x[0])
                mask = draw_line(mask, int(points_y[0]-i), points_xflip[0], i, points_x[0] ,inplace=False)
        elif j == (cols-width_2):
            for i in range(j-width_2,j):
                #print('POINTS2: ', int(points_y[0]-i), ' - ', points_xflip[0], ' - ',i, ' - ', points_x[0])
                mask = draw_line(mask, int(points_y[0]-i), points_xflip[0], i, points_x[0] ,inplace=False)
        else:
            for i in range(j-width_2,j+width_1):
                if not(int(points_y[0]-i) < 0):
                    #print('POINTS3: ', int(points_y[0]-i), ' - ', points_xflip[0], ' - ',i, ' - ', points_x[0])
                    mask = draw_line(mask, int(points_y[0]-i), points_xflip[0], i, points_x[0] ,inplace=False)




    # r = percent
    # center = [crow, ccol]
    # x, y = np.ogrid[:rows, :cols]
    # # print(np.ogrid[:rows, :cols])
    # # print(x)
    # # print(y)
    # mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    # mask[mask_area] = 1
    


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
        subsample_image(dataset_path, subsampled_path, image_to_process, 'input', spokes, name_newsubsampleimage, True)
else:
    for image_to_process in dataset_listimages:
        print('image_to_process ', image_to_process)
        subsample_image(dataset_path, subsampled_path, image_to_process, 'input', spokes, name_newsubsampleimage, False)




