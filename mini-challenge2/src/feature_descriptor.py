import cv2 as cv


def rotate_image(image, angle):
    image_rotated = image.rotate(angle)
    return image_rotated


def create_SIFT(image, grey_image):
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
