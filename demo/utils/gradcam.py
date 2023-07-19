import numpy as np
import cv2

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      bbox: tuple,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as a heatmap
    inside a specific bounding box.
    Modified from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param bbox: A tuple containing the bounding box coordinates in the format (x1, y1, x2, y2).
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay in the bbox region.
    """
    x1, y1, x2, y2 = bbox

    # Resize the heatmap to the bbox size
    heatmap_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

    # Apply the colormap to the resized heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should be np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    # Create a blank canvas to overlay the heatmap
    canvas = np.zeros_like(img)
    canvas[y1:y2, x1:x2, :] = heatmap

    # Blend the original image with the heatmap using image_weight
    cam = (1 - image_weight) * img + image_weight * canvas
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)
