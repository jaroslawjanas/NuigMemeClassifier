import math


# from https://github.com/opencv/opencv/blob/1f726e81f91746e16f4a6110681658f8709e7dd2/samples/dnn/text_detection.py#L86
# This function is used to extract the bounding box coordinates of a text region
# and the probability of a text region detection
def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    # CHECK DIMENSIONS AND SHAPES OF geometry AND scores
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
        scores_data = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        angles_data = geometry[0][4][y]
        for x in range(0, width):
            score = scores_data[x]

            # If score is lower than threshold score, move to next x
            if score < scoreThresh:
                continue

            # Calculate offset
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles_data[x]

            # Calculate cos and sin of angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offset_x + cos_a * x1_data[x] + sin_a * x2_data[x], offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

            # Find points for rectangle
            p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
            p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]
