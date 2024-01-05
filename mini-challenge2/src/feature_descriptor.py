import cv2 as cv


def rotate_image(image, angle):
    """
    Rotate an image by the specified angle.

    Args:
        image (PIL.Image.Image): The image to be rotated.
        angle (float): The angle in degrees by which the image should be rotated.

    Returns:
        PIL.Image.Image: The rotated image.
    """
    image_rotated = image.rotate(angle)
    return image_rotated


def create_SIFT(image, grey_image):
    """
    Create SIFT keypoints and descriptors for a given image.

    Args:
        image (numpy.ndarray): The original color image.
        grey_image (numpy.ndarray): The grayscale version of the original image.

    Returns:
        Tuple[numpy.ndarray, list, numpy.ndarray]: A tuple containing:
            - image (numpy.ndarray): The original color image with SIFT keypoints drawn on it.
            - keypoints (list): List of SIFT keypoints detected in the grayscale image.
            - descriptors (numpy.ndarray): SIFT descriptors for the detected keypoints.
    """
    image = image.copy()
    grey_image = grey_image.copy()
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grey_image, None)
    image = cv.drawKeypoints(image, keypoints, None)
    return image, keypoints, descriptors


def match_images_with_brisk(
    image1,
    image2,
    threshold,
    octaves,
    patternScale,
    matches_to_show=10,
    matchesThickness=5,
):
    """
    Match two images using the BRISK feature detector and descriptor extractor.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        threshold (int): Threshold for the BRISK keypoint detector.
        octaves (int): The number of octaves for the BRISK keypoint detector.
        patternScale (float): Pattern scale for the BRISK keypoint detector.
        matches_to_show (int, optional): Number of matches to visualize. Default is 10.
        matchesThickness (int, optional): Thickness of the lines used to draw matches. Default is 5.

    Returns:
        numpy.ndarray: An image showing the matches between the two input images.
    """
    brisk = cv.BRISK_create(
        thresh=threshold, octaves=octaves, patternScale=patternScale
    )

    keypoint_brisk_image1 = brisk.detect(image1, None)
    keypoint_brisk_image2 = brisk.detect(image2, None)

    keypoint_brisk_image1, descriptors_brisk_image1 = brisk.compute(
        image1, keypoint_brisk_image1
    )
    keypoint_brisk_image2, descriptors_brisk_image2 = brisk.compute(
        image2, keypoint_brisk_image2
    )

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors_brisk_image1, descriptors_brisk_image2)

    matches = sorted(matches, key=lambda x: x.distance)

    image_brisk_matches = cv.drawMatches(
        image1,
        keypoint_brisk_image1,
        image2,
        keypoint_brisk_image2,
        matches[:matches_to_show],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=matchesThickness,
    )

    return image_brisk_matches
