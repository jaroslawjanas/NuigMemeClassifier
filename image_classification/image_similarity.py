import os
import cv2

# Based on https://docs.opencv.org/4.5.2/dc/dc3/tutorial_py_matcher.html
def image_matching(img, bounding_boxes, templatePaths):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # lets cover text with white filled boxes
    # this will help with ignoring those areas
    # when extracting image descriptors
    for box in bounding_boxes:
        p1 = box[0]
        p3 = box[2]
        cv2.rectangle(img, p1, p3, (255, 255, 255), -1)

    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    orb = cv2.ORB_create(nfeatures=700)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    bfmatcher = cv2.BFMatcher()
    best_match = [0, None, None, None]
    for tempPath in templatePaths:
        tempImg = cv2.imread(tempPath, cv2.IMREAD_GRAYSCALE)
        # print(tempPath)
        # print(tempImg)
        t_keypoints, t_descriptors = orb.detectAndCompute(tempImg, None)
        if t_keypoints is None or t_descriptors is None:
            continue

        matches = bfmatcher.knnMatch(descriptors, t_descriptors, k=2)
        matching_features = []
        for a, b in matches:
            if a.distance < 0.75 * b.distance:
                matching_features.append([a])

        if len(matching_features) > best_match[0]:
            best_match[0] = len(matching_features)
            best_match[1] = tempPath
            best_match[2] = tempImg
            best_match[3] = matching_features

    if best_match[0] == 0:
        print('A matching template could not be found')
    else:
        if best_match[0] < 20:
            print('The matching template might be incorrect')

        print('Template: ' + best_match[1].split('\\').pop().split('.')[0])
        t_keypoints, t_descriptors = orb.detectAndCompute(best_match[2], None)
        img5 = cv2.drawMatchesKnn(img, keypoints, best_match[2], t_keypoints, best_match[3], None, flags=2)
        cv2.imshow('', img5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
