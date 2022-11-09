import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Union, Dict, Any, Callable

# With inspiration from https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
def blur_detector(
    image: List[List], 
    thresh: float = 8, 
    size: int = 50,
    dft: Callable = np.fft.fft2,
    idft: Callable = np.fft.ifft2) -> Dict[str, Any]:
    '''Computes a mean of any evaluation measure over a set of queries.

    Args:
        image: grayscale image to be evaluated for blur (2d np.array)
        thresh: threshold for which scores you will classsify as blurry or not
        size: half the length (in pixels) of the square used as 
            filter to remove the lower frequencies
        dft: Discrete Fourier Transform function to use
        idft: Inverse Discrete Fourier Transform function to use

    Returns:
        Dictionary with keys 'is_blurry', 'score' and (optional) 'step_images':
            is_blurry (bool): basically same as score < thresh,
            score (float): a metric of how blurry the image is,
            ret_vis (List[List[List]]): a list of matrices (images) for 
                each step in the process
    '''
    # Make sure image is 2D (Grayscale)
    assert(len(image.shape) == 2)
    h, w = image.shape
    x_center, y_center = int(w / 2.), int(h / 2.)
    # function to calculate the magnitude (0 values are left alone)
    mag = lambda G, scale=20: scale * (np.ma.log(np.abs(G))).filled(0)

    # Take the Discrete Fourier Transform of the image
    G = dft(image)
    # Shift the DC component to the middle
    G_shift = np.fft.fftshift(G)
    # filter out the lower frequencies (set them to 0)
    G_shift_filtered = copy.copy(G_shift)
    G_shift_filtered[(y_center - size) : (y_center + size),
                     (x_center - size) : (x_center + size)] = 0
    # Shift the DC component back to the top left corner
    G_filtered = np.fft.ifftshift(G_shift_filtered)
    recon = idft(G_filtered)
    recon_mag = mag(recon)
    score = np.mean(recon_mag)
    
    # Images that are to be returned for visualization
    step_images = [
        image,                   # Original image
        mag(G_shift),            # Shifted magnitude
        mag(G_shift_filtered),   # Filtered shifted magnitude
        np.real(recon),          # The real part of the recon. image
        mag(recon),              # Magnnitude of the recon. image
    ]

    return {
        'is_blurry': score < thresh,
        'score': score,
        'step_images': step_images
    }

def plot_images(images, labels=['original image', 'magnitude', 
                                'magnitude-filtered', 'reconstructed image', 
                                'magnitude of the reconstructed image']):
    frec_scale = (images[1].min(), images[1].max())
    plt.figure(figsize=(12, 12))
    plt.suptitle('Step by step in the blur_detection algorithm')
    for i, img in enumerate(images):
        plt.subplot(int(f'23{i+1}'))
        plt.title(str(labels[i]))
        if i == 2:
            plt.imshow(img, cmap='gray', vmin=frec_scale[0], vmax=frec_scale[1])
        else:
            plt.imshow(img, cmap='gray')
        plt.tick_params(left=False, right=False , labelleft=False,
                        labelbottom=False, bottom=False)

def main():
    # Load image
    lenna = cv2.imread('data/images/lenna.png', flags=cv2.IMREAD_GRAYSCALE)
    # a threshold of 8 and a size of 50 seems to work well for a lot of images
    r = blur_detector(lenna, thresh=8, size=50)
    # Plot images for each step in the process
    plot_images(r['step_images'])
    plt.show()
    # Print results to terminal
    print(f'is_blurry: {r["is_blurry"]}')
    print(f'score: {r["score"]}')

if __name__ == '__main__':
    main()