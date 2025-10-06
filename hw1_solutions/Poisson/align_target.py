import numpy as np
import cv2
from get_mask import get_mask


def click_event(event, x, y, flags, params):
    t_lst = params['t_lst']
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        t_lst[1] = x
        t_lst[0] =  y


def align_target(source_image, target_image):
    mask = get_mask(source_image.copy())

    source_h, source_w = target_image.shape[:2]
    ty, tx = source_h / 2., source_w / 2.

    scale = 1.
    angle = 0.
    while True:
        source_copy = source_image.copy()
        M = cv2.getRotationMatrix2D((source_copy.shape[1] / 2, source_copy.shape[0] / 2), angle, scale)
        source_copy = cv2.warpAffine(source_copy, M, (source_copy.shape[1], source_copy.shape[0]))
        mask_copy = mask.copy()
        mask_copy = cv2.warpAffine(mask_copy, M, (mask_copy.shape[1], mask_copy.shape[0]))
        mask_pixels = np.argwhere(mask_copy == 1)
        y = mask_pixels[:, 0]
        x = mask_pixels[:, 1]
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        target_copy = target_image.copy()
        yind2 = np.clip(np.arange(ymin, ymax) - int(y.mean()) + int(ty), 0, target_copy.shape[0] - 1).astype(int)
        yind = yind2 + int(y.mean()) - int(ty)
        xind2 = np.clip(np.arange(xmin, xmax) - int(x.mean()) + int(tx), 0, target_copy.shape[1] - 1).astype(int)
        xind = xind2 + int(x.mean()) - int(tx)

        y = np.clip(y - int(y.mean()) + int(ty), 0, target_copy.shape[0] - 1).astype(int)
        x = np.clip(x - int(x.mean()) + int(tx), 0, target_copy.shape[1] - 1).astype(int)

        mask2 = np.zeros((target_copy.shape[0], target_copy.shape[1]), dtype=np.float32)
        mask2[y, x] = 1

        yy, xx = np.meshgrid(yind, xind, indexing='ij')
        yy2, xx2 = np.meshgrid(yind2, xind2, indexing='ij')
        ind = np.stack((yy, xx), axis=2).reshape(-1, 2)
        ind2 = np.stack((yy2, xx2), axis=2).reshape(-1, 2)

        im_s2 = np.zeros_like(target_copy)
        im_s2[ind2[:, 0], ind2[:, 1]] = source_copy[ind[:, 0], ind[:, 1]]

        target_copy[mask2 == 1] = im_s2[mask2 == 1]
        cv2.imshow('target cloned', target_copy)
        t_lst = [ty, tx]
        # key = cv2.waitKey(0)
        key = cv2.waitKeyEx(0)
        if key & 0xFF == ord('r'):
            angle += 10.
        elif key & 0xFF == ord('s'):
            scale += 0.1
        elif key & 0xFF == ord('a'):
            scale -= 0.1
        elif key == 0 or key == 2490368:
            ty -= 5
        elif key == 1 or key == 2621440:
            ty += 5
        elif key == 2 or key == 2424832:
            tx -= 5
        elif key == 3 or key == 2555904:
            tx += 5
        elif key & 0xFF == ord('q'):
            break
    return target_copy, mask2



