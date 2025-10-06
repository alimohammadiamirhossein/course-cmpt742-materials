import numpy as np
import cv2


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    source_image = params['source_image']
    xs = params['xs']
    ys = params['ys']
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(source_image, (x, y), 3, (255, 0, 0), -1)
        if len(xs) > 1:
            cv2.line(source_image, (xs[-2], ys[-2]), (xs[-1], ys[-1]), (0, 0, 255), 1)
        cv2.imshow('image', source_image)



def get_mask(source_image):
    cv2.imshow('image', source_image)
    xs = []
    ys = []


    cv2.setMouseCallback('image', click_event, {'source_image': source_image, 'xs': xs, 'ys': ys})
    cv2.waitKey(0)

    xs = np.array(xs)
    ys = np.array(ys)

    mask = np.zeros((source_image.shape[0], source_image.shape[1]), dtype=np.float32)
    mask = cv2.fillPoly(mask, [np.stack((xs, ys), axis=1)], 1)
    return mask


