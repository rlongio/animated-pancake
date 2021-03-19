import cv2
import numpy as np


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """maintain_aspect_ratio_resize returns a resized image that maintains the aspect ratio"""
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def load_template(file_path):
    """load_template loads and prepares an image file to be used as a template
    when templateMatching

    Args:
        file_path (str): path to the file
    """
    # Template
    template = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    return template


def auto_canny(image, sigma=0.33):
    """auto_canny find the optimal edging

    From: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    Args:
        image ([type]): [description]
        sigma (float, optional): [description]. Defaults to 0.33.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def get_x_y_coordinates(tH, tW, found):
    """get_x_y_coordinates returns coordinates generated from templateMatch

    Args:
        tH (int): Height
        tW (int): Width
        found (Any): result of matchTemplate
    """
    (_, max_loc, r) = found
    (start_x, start_y) = (int(max_loc[0] * r), int(max_loc[1] * r))
    (end_x, end_y) = (int((max_loc[0] + tW) * r), int((max_loc[1] + tH) * r))
    return (start_x, start_y, end_x, end_y)


def get_circle_coords_from_square(tW, iW, start_x, end_x):
    """get_circle_coords_from_square returns info needed to draw a circle
    from a set of square coords
    """
    x = start_x + round(iW * 0.5)
    y = end_x + round(iW * 0.5)
    r = tW / 2
    H = round(abs(start_x - end_x) / 2)
    W = round(abs(start_y - end_y) / 2)
    return (x, y), r, (H, W)


def show_search_result(original_image, start_x, start_y, end_x, end_y):
    """show_search_result convenience for showing the user the match result
    from a matchTemplate
    """
    print("display search result to user")
    image_copy = original_image.copy()
    cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
    cv2.imshow("og", image_copy)


def write_image(original_image, iH, iW, x, y):
    """write_image writes the image to disk"""
    circleIn = np.zeros((iH, iW, 1), np.uint8)
    circleIn = cv2.circle(
        circleIn,
        (x, y),
        min(iH, iW),
        (1),
        -1,
    )
    circleOut = circleIn.copy()
    circleOut[circleOut == 0] = 2
    circleOut[circleOut == 1] = 0
    circleOut[circleOut == 2] = 1
    # Generate a blank image
    imgIn = np.zeros((iH, iW, 4), np.uint8)
    # Copy the first 3 channels
    imgIn[:, :, 0] = np.multiply(original_image[:, :, 0], circleIn[:, :, 0])
    imgIn[:, :, 1] = np.multiply(original_image[:, :, 1], circleIn[:, :, 0])
    imgIn[:, :, 2] = np.multiply(original_image[:, :, 2], circleIn[:, :, 0])
    # Set the opaque part of the alpha channelcircleIn[circleIn == 1] = 255
    imgIn[:, :, 3] = circleIn[:, :, 0]
    cv2.imshow("./output/imgIn.png", imgIn)
    cv2.imwrite("./output/imgIn.png", imgIn)


if __name__ == "__main__":

    template = load_template("./templates/template3.png")
    (tH, tW) = template.shape[:2]

    original_image = cv2.imread("./input/test3.jpg", cv2.IMREAD_UNCHANGED).copy()
    (iH, iW) = original_image.shape[:2]

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = maintain_aspect_ratio_resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        auto = auto_canny(blurred)
        detected = cv2.matchTemplate(auto, template, cv2.TM_CCOEFF)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(detected)

        if found is None or max_val > found[0]:
            found = (max_val, max_loc, r)

    if found is None:
        print(f"no match found {None}")
        exit(1)

    start_x, start_y, end_x, end_y = get_x_y_coordinates(tH, tW, found)
    (x, y), r, (fH, fW) = get_circle_coords_from_square(tW, iW, start_x, end_x)

    show_search_result(original_image, start_x, start_y, end_x, end_y)

    write_image(original_image, iH, iW, x, y)

    cv2.waitKey(0)
    exit()
