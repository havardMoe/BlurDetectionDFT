import numpy as np
import math
import numpy.typing as npt

def DFT_1D(g: npt.ArrayLike) -> npt.ArrayLike:
    ''' Function for calculating the discrete fourier transform in 1D
    Parameters:
        g(np.array): 1D signal
    Returns:
        G(np.array): Frequency domain
    '''
    G_even = [0 for i in range(len(g))]
    G_odd = [0 for i in range(len(g))]
    w = len(g)
    for k in range(w):
        f = k/w
        for x in range(w):
            G_even[k] = G_even[k] + g[x]*math.cos(2*np.pi*f*x)
            G_odd[k] = G_odd[k] - g[x]*math.sin(2*np.pi*f*x)
    return np.array([complex(g_ev, g_odd) for g_ev, g_odd in zip(G_even, G_odd)])

def DFTI_1D(G: npt.ArrayLike) -> npt.ArrayLike:
    ''' Function for calculating the inverse discrete fourier transform in 1D
    Parameters:
        G(np.array): Frequency domain -> 1D array
    Returns:
        g(np.array): Time domain -> 1D array
    '''
    w = len(G)
    g_real = [0 for i in range(w)]

    for x in range(w):
        for k in range(w):
            f = k/w
            # Exponential part with eulers rule for cos and sin
            ex = (math.cos(2*np.pi*f*x)+1j*math.sin(2*np.pi*f*x))
            g_real[x] += ex*G[k]

    return np.array([(1/w)*g for g in g_real])

def DFT_2D(g_img: npt.NDArray) -> npt.NDArray:
    ''' Function for calculating the discrete fourier transform in 1D
    Parameters:
        g(np.array): 2D Image
    Returns:
        G(np.array): Frequency domain 2D
    '''

    w,h = g_img.shape
    G = np.zeros((w,h), dtype = 'complex_')
    temp_G = np.zeros((w,h), dtype = 'complex_')
    for y in range(h):
        temp_G[:,y] = DFT_1D(g_img[:,y])
        
    for x in range(w):
        G[x,:] = DFT_1D(temp_G[x,:])
    return G

def DFTI_2D(G: npt.NDArray) -> npt.NDArray:
    ''' Function for calculating the inverse discrete fourier transform in 2D
    Parameters:
        G(np.NDArray): Frequency domain -> 2D NDA
    Returns:
        g(np.NDArray): Time domain -> 1D array
    '''
    w, h = G.shape
    g = np.zeros((w,h), dtype = 'complex_')
    temp_g = np.zeros((w,h), dtype = 'complex_')
    for y in range(h):
        temp_g[:,y] = DFTI_1D(G[:,y])
    for x in range(w):
        g[x,:] = DFTI_1D(temp_g[x,:])
    return g