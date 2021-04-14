# for neighbouring boxes
import math


def box_grouping(bounding_boxes, x_range, y_range):
    grouped_bounding_boxes = bounding_boxes[:]
    merge_occurred = True
    while merge_occurred:
        merge_occurred = False
        boxes_to_be_grouped = find_boxes_to_be_grouped(grouped_bounding_boxes, x_range, y_range)
        if boxes_to_be_grouped is not None:
            box1, box2 = boxes_to_be_grouped
            merge_occurred = True
            grouped_bounding_boxes.remove(box1)
            grouped_bounding_boxes.remove(box2)
            grouped_bounding_boxes.append(box_merge(box1, box2))

    return grouped_bounding_boxes


def find_boxes_to_be_grouped(bounding_boxes, x_range, y_range):
    closest_box = {
        'distance': 99999999,
        'box1': None,
        'box2': None
    }
    collision_box = {
        'box1': None,
        'box2': None
    }
    # loop bounding boxes
    for box1 in bounding_boxes[:]:
        # for each point of the box
        for p in box1:
            # loop through all bounding boxes
            for box2 in bounding_boxes[:]:
                if box1 == box2:
                    continue
                # for each point of box2
                for pp in box2:
                    # distance merge
                    x_distance = abs(p[0] - pp[0])
                    y_distance = abs(p[1] - pp[1])
                    # boxes satisfying this condition should be merged
                    # in order from smallest to biggest distance
                    if x_distance < x_range and y_distance < y_range:
                        dist = math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance, 2))
                        if dist < closest_box['distance']:
                            closest_box['distance'] = dist
                            closest_box['box1'] = box1
                            closest_box['box2'] = box2

                    # if box collision - this will be dealt with only when
                    # there are no more closest boxes to join
                    if box1[0][0] < pp[0] < box1[2][0] and box1[0][1] < pp[1] < box1[2][1]:
                        collision_box['box1'] = box1
                        collision_box['box2'] = box2

    if closest_box['box1'] is not None and closest_box['box2'] is not None:
        return [closest_box['box1'], closest_box['box2']]
    # only do box collision if there are no closest boxes left to join
    elif collision_box['box1'] is not None and collision_box['box2'] is not None:
        return [collision_box['box1'], collision_box['box2']]

    return None


# function to merge two boxes
def box_merge(box1, box2):
    # find the biggest and smallest [x, y] values
    x_min = min(box1[0][0], box2[0][0])
    x_max = max(box1[2][0], box2[2][0])
    y_min = min(box1[0][1], box2[0][1])
    y_max = max(box1[2][1], box2[2][1])

    p1 = (x_min, y_min)
    p2 = (x_max, y_min)
    p3 = (x_max, y_max)
    p4 = (x_min, y_max)

    return [p1, p2, p3, p4]


def point_distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


# for boxes whose vertices are contained in other boxes
def box_collision(bounding_boxes):
    return 0
