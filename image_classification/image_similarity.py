import os
import cv2


# Based on https://docs.opencv.org/4.5.2/dc/dc3/tutorial_py_matcher.html
def image_matching(img, bounding_boxes, templates, config):
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
    input_img_key_points, input_img_descriptors = orb.detectAndCompute(img, None)

    bf_matcher = cv2.BFMatcher()
    best_match = {
        'matching_features': [],
        'template_name': None,
        'template_img': None,
    }

    for template_name in templates:
        template_dir_path = os.path.join(config['data']['templates']['path'], template_name)
        for file_name in os.listdir(template_dir_path):
            file_path = os.path.join(template_dir_path, file_name)

            template_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            template_key_points, template_descriptors = orb.detectAndCompute(template_img, None)
            if template_key_points is None or template_descriptors is None:
                continue

            matches = bf_matcher.knnMatch(input_img_descriptors, template_descriptors, k=2)
            matching_features = []
            for a, b in matches:
                if a.distance < 0.75 * b.distance:
                    matching_features.append([a])

            if len(matching_features) > len(best_match['matching_features']):
                best_match['matching_features'] = matching_features
                best_match['template_name'] = template_name
                best_match['template_img'] = template_img

    if len(best_match['matching_features']) == 0:
        print('A matching template could not be found')
    else:
        if len(best_match['matching_features']) < 20:
            print('The matching template might be incorrect')

        print('Template: ' + best_match['template_name'])
        template_key_points, template_descriptors = orb.detectAndCompute(best_match['template_img'], None)
        img5 = cv2.drawMatchesKnn(img, input_img_key_points, best_match['template_img'],
                                  template_key_points, best_match['matching_features'], None, flags=2)
        cv2.imshow('', img5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
