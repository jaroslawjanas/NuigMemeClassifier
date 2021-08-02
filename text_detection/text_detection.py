import cv2
import os
from text_detection.decode_bounding_boxes import *
from text_detection.box_grouping import *


# Based on https://stackoverflow.com/questions/54821969/how-to-make-bounding-box-around-text-areas-in-an-image-even-if-text-is-skewed
# and https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def text_detection(raw_image, config):
    # params
    conf_threshold = config['text_detection']['confidence_threshold'] #0.63
    # Non Max Suppression
    # Higher value will result in more boxes - suppresses weak overlapping bounding boxes
    # https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    # https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/
    nms_threshold = config['text_detection']['nms_threshold'] #0.77

    # image properties
    height, width = raw_image.shape[:2]
    dimensions = (width, height)

    # Display image
    cv2.imshow('Original Image', raw_image)
    print('Original dimensions: ', dimensions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate new dimensions (multiple of 32, required by EAST detector)
    new_width = math.floor(width / 32) * 32
    new_height = math.floor(height / 32) * 32
    new_dimensions = (new_width, new_height)

    # Image resize
    image_resized = cv2.resize(raw_image, new_dimensions)
    print('New dimensions: ', new_dimensions)

    # Load the NN model
    # https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
    dir_path = os.path.dirname(__file__)
    nn_model_path = os.path.join(dir_path, 'text_detection_nnModel/frozen_east_text_detection.pb')
    net = cv2.dnn.readNet(nn_model_path)

    # prepare image input - convert to 4D blob
    img_blob = cv2.dnn.blobFromImage(image_resized, 1.0, new_dimensions, (123.68, 116.78, 103.94), True, False)

    # Output layers, text geometry and confidence
    output_layers = []
    output_layers.append("feature_fusion/Conv_7/Sigmoid")
    output_layers.append("feature_fusion/concat_3")

    # Forward Feed
    net.setInput(img_blob)
    output = net.forward(output_layers)
    # geometry - map used to derive the bounding box coordinates of text in the image
    # scores = map, containing the probability of a given region containing text
    scores, geometry = output[:2]  # unpack

    # boxes = shapes with text in them
    # confidences = level of confidence that a box contains text
    [boxes, confidences] = decodeBoundingBoxes(scores, geometry, conf_threshold)
    # these are the indices of the boxes that we want to keep after thresholding
    box_indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)

    # calculate ratio of dimensions - since we want to display bounding boxes from a smaller image
    # on the original bigger image
    rW = width / float(new_width)
    rH = height / float(new_height)

    # loop through kept boxes
    bb_image = raw_image.copy()
    bounding_boxes = []
    for i in box_indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])

        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] = round(vertices[j][0] * rW)  # rescale width
            if vertices[j][0] < 0:
                vertices[j][0] = 0
            elif vertices[j][0] > width:
                vertices[j][0] = width

            vertices[j][1] = round(vertices[j][1] * rH)  # rescale height
            if vertices[j][1] < 0:
                vertices[j][1] = 0
            elif vertices[j][1] > height:
                vertices[j][1] = height


        # construct bounding boxes
        #  p1------------------p2
        #  |                   |
        #  |                   |
        #  p4------------------p3
        p1 = (int(round(vertices[1][0])), int(round(vertices[1][1])))
        p2 = (int(round(vertices[2][0])), int(round(vertices[2][1])))
        p3 = (int(round(vertices[3][0])), int(round(vertices[3][1])))
        p4 = (int(round(vertices[0][0])), int(round(vertices[0][1])))
        bounding_boxes.append([p1, p2, p3, p4])

        # draw all bounding boxes
        cv2.rectangle(bb_image, p1, p3, (255, 170, 0), 2)

    # Display image
    cv2.imshow('Text Detection', bb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # group bounding boxes
    # boxes, x-range, y-range
    grouped_bounding_boxes = box_grouping(bounding_boxes, width/10, height/10)

    grouped_boxes_img = raw_image.copy()
    for box in grouped_bounding_boxes:
        cv2.rectangle(grouped_boxes_img, box[0], box[2], (255, 170, 0), 2)

    # show image with bounding boxes
    cv2.imshow('Grouped Text Detection', grouped_boxes_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grouped_bounding_boxes
