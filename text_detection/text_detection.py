import cv2
import math
import os

# from https://github.com/opencv/opencv/blob/1f726e81f91746e16f4a6110681658f8709e7dd2/samples/dnn/text_detection.py#L86
# This function is used to extract the bounding box coordinates of a text region
# and the probability of a text region detection
def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


# Based on https://stackoverflow.com/questions/54821969/how-to-make-bounding-box-around-text-areas-in-an-image-even-if-text-is-skewed
def image_processing(img_path):
    # params
    conf_threshold = 0.65
    # Non Max Suppression
    # Higher value will result in more boxes
    # https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    # https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/
    nms_threshold = 0.2

    # Read image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    dimensions = (width, height)

    # Display image
    cv2.imshow('Original Image', image)
    print('Original dimensions: ', dimensions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate new dimensions (multiple of 32, required by EAST detector)
    new_width = math.floor(width/32)*32
    new_height = math.floor(height/32)*32
    new_dimensions = (new_width, new_height)

    # Display new image
    image_resized = cv2.resize(image, new_dimensions)
    cv2.imshow('Resized mod(32)px image', image_resized)
    print('New dimensions: ', new_dimensions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load the NN model
    # https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
    dirpath = os.path.dirname(__file__)
    nn_model_path = os.path.join(dirpath, 'text_detection_nnModel/frozen_east_text_detection.pb')
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
    bounding_boxes = []
    for i in box_indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])

        # scale the bounding box coordinates based on the respective ratios

        for j in range(4):
            vertices[j][0] *= rW  # rescale width
            vertices[j][1] *= rH  # rescale height

        # construct bounding boxes
        p1 = (vertices[0][0], vertices[0][1])
        p2 = (vertices[2][0], vertices[2][1])
        bounding_boxes.append((p1, p2))
        # cv2.line(image, p1, p2, (0, 255, 0), 3)
        cv2.rectangle(image, p1, p2, (255, 170, 0), 1)

    # Display image
    cv2.imshow('Text Detection Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()