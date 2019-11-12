import numpy as np


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (intersection_area / float(bb1_area + bb2_area - intersection_area))*bb1_area/bb2_area
    return iou

def NMS(boxes, scores, labels, THRESH=.5, IoU_trash=.99):
    not_ok_id = []
    if scores[0].__len__():
        for elem in range(scores[0].__len__()):
            if scores[0][elem] > THRESH:
                for box_b in range(boxes[0].__len__()):
                    if scores[0][box_b] > THRESH:
                        if elem != box_b:
                            if get_iou(boxes[0][elem], boxes[0][box_b]) > IoU_trash:
                                if scores[0][elem] > scores[0][box_b]:
                                    not_ok_id.append(box_b)
                                else:
                                    not_ok_id.append(elem)
                    else: not_ok_id.append(box_b)
            else: not_ok_id.append(elem)
        not_ok_id = np.unique(not_ok_id)
        ok_id = []
        for elem in range(boxes[0].__len__()):
            if elem not in not_ok_id:
                ok_id.append(elem)
        boxes_ = []
        boxes_.append(np.array([boxes[0][i] for i in ok_id]))
        boxes = np.array(boxes_)
        classes_ = []
        classes_.append(np.array([labels[0][i] for i in ok_id]))
        labels = np.array(classes_)
        scores_ = []
        scores_.append(np.array([scores[0][i] for i in ok_id]))
        scores = np.array(scores_)
    return boxes, scores, labels
