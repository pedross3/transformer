# util_images.py
import numpy as np
import cv2

def homography_derotation(K_src, K_trg, R):
    """
    Compute homography to derotate target into source-aligned frame:
      H = K_src @ R.T @ inv(K_trg)
    K_src, K_trg: (3,3)
    R: (3,3) rotation from src -> trg (so R transforms src coords to trg coords).
    Returns H (3x3) suitable for cv2.warpPerspective.
    """
    H = K_src.dot(R.T).dot(np.linalg.inv(K_trg))
    return H

def derotate_image_cv2(img, H, out_size):
    """
    img: HxWxC uint8/float32
    H: 3x3 homography
    out_size: (width, height) for cv2.warpPerspective
    """
    # cv2 expects (w,h)
    warped = cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return warped


