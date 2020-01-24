def calculate_iou(predicted_box, groundtruth_box):
    """
    Calculates the "intersection over union" metric for object detection.

    The metric indicates how well a detected bounding box aligns with the
    real bounding box. If two bounding boxes align very well the IoU should
    be 1.0. If they do not align at all, the IoU will be 0.0.

    :param predicted_box: The [x_min, y_min, x_max, y_max] coordinates of
                          the predicted bounding box.
    :param groundtruth_box: The [x_min, y_min, x_max, y_max] coordinates of
                            the groundtruth bounding box.

    :return: A float value between 0.0 and 1.0.
    """
    intersection_min_x = max(predicted_box[0], groundtruth_box[0])
    intersection_min_y = max(predicted_box[1], groundtruth_box[1])
    intersection_max_x = min(predicted_box[2], groundtruth_box[2])
    intersection_max_y = min(predicted_box[3], groundtruth_box[3])

    intersection_width = max(0, intersection_max_x - intersection_min_x + 1)
    intersection_height = max(0, intersection_max_y - intersection_min_y + 1)
    intersection_area = intersection_width * intersection_height

    predicted_width = (predicted_box[2] - predicted_box[0] + 1)
    predicted_height = (predicted_box[3] - predicted_box[1] + 1)
    predicted_area = predicted_width * predicted_height

    groundtruth_width = (groundtruth_box[2] - groundtruth_box[0] + 1)
    groundtruth_height = (groundtruth_box[3] - groundtruth_box[1] +1)
    groundtruth_area = groundtruth_width * groundtruth_height

    union_area = groundtruth_area + predicted_area - intersection_area

    return intersection_area / union_area
