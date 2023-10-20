import os
import numpy as np
from math import ceil, floor

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

def check_diff(x, y) -> None:
        eps = 1e-3
        diff = torch.norm(x - y, 2)
        # diff = np.linalg.norm(x - y, 2)
        if diff > eps:
            print('Implementation:')
            # print(x)
            print('MATLAB reference:')
            # print(y)
            raise ArithmeticError(
                'Difference is not negligible!: {}'.format(diff),
            )
        else:
            print('Allowable difference: {:.4e} < {:.4e}'.format(diff, eps))

        return

if __name__ == "__main__":
    from PIL import Image
    import torch

    import numpy as np
    from skimage.io import imsave, imread
    from skimage import img_as_float

    from basicsr.utils import FileClient, imfrombytes

    # hr_img_path = "/mnt/aigc_cq/shared/super_resolution/DIV2K_NTIRE22_Learning_SR_Space/DIV2K-tr_1X/0101.png"

    # # 1. Load original image and do imresize+save both im UINT8 and FLOAT64 types
    # # ori
    # img_uint8 = imread(hr_img_path)
    
    # # my read
    # file_client = FileClient('disk')
    # gt_img_bytes = file_client.get(hr_img_path)
    # img_gt = imfrombytes(gt_img_bytes, float32=False)
    
    # print((img_gt[:,:,::-1]-img_uint8==0).all())


    # new_size = (int(img_uint8.shape[0]/8), int(img_uint8.shape[1]/8))
    # new_img_uint8 = imresize(img_uint8, output_shape=new_size)
    # # imsave('py_lena_123x234_uint8.png', new_img_uint8)
    # img_double = img_as_float(img_uint8)
    # new_img_double = imresize(img_double, output_shape=new_size)
    # imsave('py_lena_123x234_double.png', convertDouble2Byte(new_img_double))

    # # 2. Load images resized by python's imresize() and compare with images resized by MatLab's imresize()
    # matlab_uint8 = imread('/mnt/aigc_cq/shared/super_resolution/DIV2K_NTIRE22_Learning_SR_Space/DIV2K-tr_8X/0101.png')
    # # python_uint8 = imread('py_lena_123x234_uint8.png')
    # matlab_double = imread('/mnt/aigc_cq/shared/super_resolution/DIV2K_NTIRE22_Learning_SR_Space/DIV2K-tr_8X/0101.png')
    # python_double = imread('py_lena_123x234_double.png')
    # # diff_uint8 = matlab_uint8.astype(np.int32) - python_uint8.astype(np.int32)
    
    # check_diff(torch.Tensor(matlab_uint8), torch.Tensor(new_img_uint8))

    # diff_uint8 = matlab_uint8.astype(np.int32) - new_img_uint8.astype(np.int32)
    # diff_double = matlab_double.astype(np.int32) - python_double.astype(np.int32)
    # # print('Python/MatLab uint8 diff: min =', np.amin(diff_uint8), 'max =', np.amax(diff_uint8))

    # print('Python/MatLab uint8 diff: min =', np.amin(diff_uint8), 'max =', np.amax(diff_uint8))

    # print('Python/Matlab double diff: min =', np.amin(diff_double), 'max =', np.amax(diff_double))

    # # **** Final conclusion: DIV2K使用的是matlab_unit8, 可以用imresize代替 *****



    # 检查所有DIV2K的数据
    train_path = "/mnt/aigc_cq/shared/super_resolution/DIV2K_NTIRE22_Learning_SR_Space/DIV2K-tr_1X/"
    fns = os.listdir(train_path)
    for fn in fns:
        file_client = FileClient('disk')
        gt_img_bytes = file_client.get(os.path.join(train_path, fn))
        img_gt = imfrombytes(gt_img_bytes, float32=False)[:,:,::-1]

        # img_uint8 = imread(os.path.join(train_path, fn))

        new_size = (int(img_gt.shape[0]/8), int(img_gt.shape[1]/8))
        new_img_uint8 = imresize(img_gt, output_shape=new_size)
        matlab_uint8 = imread(os.path.join(train_path, fn).replace("1X", "8X"))
        print((new_img_uint8-matlab_uint8).sum())



