import numpy as np
import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    """
    Ultralytics-style letterbox: resize with unchanged aspect ratio + padding.
    Returns: padded image, scale ratio, (dw, dh) padding.
    """
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im_padded, r, (left, top)


def xywhr_to_poly(cx, cy, w, h, angle_rad):
    """
    Convert (cx,cy,w,h,angle[radians]) -> 4 corner points (x1,y1,...,x4,y4)
    angle is radians (Ultralytics OBB uses radians in xywhr). :contentReference[oaicite:3]{index=3}
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    # rectangle corners around origin (clockwise)
    half_w, half_h = w / 2.0, h / 2.0
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ], dtype=np.float32)

    rotated = corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated.reshape(-1)  # (8,)