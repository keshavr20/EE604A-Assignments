import cv2
import numpy as np

def bilateralfilter(image, texture,scale):
    w1,h1=image.shape[:2]
    image = cv2.resize(image,(int(w1*scale),int(h1*scale)))
    texture = cv2.resize(texture,(int(w1*scale),int(h1*scale)))
    r = 6
    h, w = image.shape[:2]
    I = np.pad(image, ((r, r), (r, r), (0, 0)) if image.ndim == 3 else ((r, r), (r, r)), 'reflect').astype(np.float32)
    T = np.pad(texture, ((r, r), (r, r), (0, 0)) if texture.ndim == 3 else ((r, r), (r, r)), 'reflect').astype(np.int32)

    output = np.zeros_like(image)
    scaleFactor_s = 1 / 18.0
    scaleFactor_r = 1 / 450.0
    LUT = np.exp(-np.arange(256)**2 * scaleFactor_r)
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x**2 + y**2) * scaleFactor_s)

    for y in range(r, r + h):
        for x in range(r, r + w):
            sliceI = I[y - r:y + r + 1, x - r:x + r + 1]
            sliceT = T[y - r:y + r + 1, x - r:x + r + 1]

            if image.ndim == 2 or (image.ndim == 3 and texture.ndim == 2):
                wgt = LUT[abs(sliceT - T[y, x])] * kernel_s
                if image.ndim == 2:
                    output[y - r, x - r] = (wgt * sliceI).sum() / wgt.sum()
                else:
                    wacc = wgt.sum()
                    for c in range(3):
                        output[y - r, x - r, c] = (wgt * sliceI[..., c]).sum() / wacc
            elif image.ndim == 3 and texture.ndim == 3:
                wgt = np.prod([LUT[abs(sliceT[..., i] - T[y, x, i])] for i in range(3)], axis=0) * kernel_s
                wacc = wgt.sum()
                for c in range(3):
                    output[y - r, x - r, c] = (wgt * sliceI[..., c]).sum() / wacc
    output = cv2.resize(output,(h1,w1))

    return output

def bilateralfilter1(image,scale):
    w1,h1=image.shape[:2]
    image = cv2.resize(image,(int(w1*scale),int(h1*scale)))

    r = 20
    h, w = image.shape[:2]

    # Efficient padding
    I = np.pad(image, ((r, r), (r, r), (0, 0)) if image.ndim == 3 else ((r, r), (r, r)), 'edge').astype(np.float32)
    T = np.pad(image, ((r, r), (r, r), (0, 0)) if image.ndim == 3 else ((r, r), (r, r)), 'edge').astype(np.int32)

    # Pre-compute
    output = np.zeros_like(image)
    scaleFactor_s = 1 / 200.0
    scaleFactor_r = 1 / 800.0
    LUT = np.exp(-np.arange(256)**2 * scaleFactor_r)
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x**2 + y**2) * scaleFactor_s)

    # Vectorization of the bilateral filter
    for y in range(r, r + h):
        for x in range(r, r + w):
            sliceI = I[y - r:y + r + 1, x - r:x + r + 1]
            sliceT = T[y - r:y + r + 1, x - r:x + r + 1]

            if image.ndim == 2 or (image.ndim == 3 and image.ndim == 2):
                wgt = LUT[abs(sliceT - T[y, x])] * kernel_s
                if image.ndim == 2:
                    output[y - r, x - r] = (wgt * sliceI).sum() / wgt.sum()
                else:
                    wacc = wgt.sum()
                    for c in range(3):
                        output[y - r, x - r, c] = (wgt * sliceI[..., c]).sum() / wacc
            elif image.ndim == 3 and image.ndim == 3:
                wgt = np.prod([LUT[abs(sliceT[..., i] - T[y, x, i])] for i in range(3)], axis=0) * kernel_s
                wacc = wgt.sum()
                for c in range(3):
                    output[y - r, x - r, c] = (wgt * sliceI[..., c]).sum() / wacc
    output = cv2.resize(output,(h1,w1))

    return output

def solution(image_path_a, image_path_b):
    scale = 0.4
    noflash = cv2.imread(image_path_a)
    flash = cv2.imread(image_path_b)
    img_1 = bilateralfilter(noflash, flash,scale)
    img_bf1 = bilateralfilter1(flash,scale)
    f=(flash+1e-6)/(img_bf1+1e-6)
    finalimage=(img_1*f).astype(np.int32)
    return finalimage
