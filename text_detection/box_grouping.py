# for neighbouring boxes
import math


def box_grouping(bounding_boxes, x_range, y_range):
    all_bounding_boxes = bounding_boxes[:]
    merge_occurred = True
    while merge_occurred:
        merge_occurred = False
        boxes_to_be_grouped = find_boxes_to_be_grouped(all_bounding_boxes, x_range, y_range)
        # [print(box) for box in boxes_to_be_grouped]
        # print('---------------')
        if boxes_to_be_grouped is not None:
            for box in boxes_to_be_grouped:
                all_bounding_boxes.remove(box)
            # [all_bounding_boxes.remove(box) for box in boxes_to_be_grouped]
            all_bounding_boxes.append(box_group(boxes_to_be_grouped))
            merge_occurred = True

    return all_bounding_boxes


def find_boxes_to_be_grouped(bounding_boxes, x_range, y_range):
    boxes_to_be_grouped = []
    # loop through all bounding boxes
    for box1 in bounding_boxes:
        # loop through all bounding boxes
        for box2 in bounding_boxes:
            add_this_box = False
            if box1 == box2:
                continue
            # for each point of box
            for p in box1:
                # for each point of box2
                for pp in box2:
                    # distance
                    x_distance = abs(p[0] - pp[0])
                    y_distance = abs(p[1] - pp[1])
                    # boxes satisfying this condition should be grouped
                    if (x_distance < x_range and y_distance < y_range) or box_collision(box1, pp):
                        add_this_box = True

            if add_this_box is True:
                if not boxes_to_be_grouped:
                    boxes_to_be_grouped.append(box1)
                boxes_to_be_grouped.append(box2)

        if boxes_to_be_grouped:
            return boxes_to_be_grouped

    return None


# function to merge two boxes
def box_group(boxes_to_be_grouped):
    all_x_values = []
    all_y_values = []

    for box in boxes_to_be_grouped:
        for point in box:
            all_x_values.append(point[0])
            all_y_values.append(point[1])

    x_min = min(all_x_values)
    x_max = max(all_x_values)
    y_min = min(all_y_values)
    y_max = max(all_y_values)

    p1 = (x_min, y_min)
    p2 = (x_max, y_min)
    p3 = (x_max, y_max)
    p4 = (x_min, y_max)

    return [p1, p2, p3, p4]


def point_distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


# for boxes whose vertices are contained in other boxes
def box_collision(box1, point):
    return box1[0][0] < point[0] < box1[2][0] and box1[0][1] < point[1] < box1[2][1]
