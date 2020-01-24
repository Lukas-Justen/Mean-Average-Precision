import unittest

from src.iou import calculate_iou


class IouTestCase(unittest.TestCase):

    def test_indentical_area(self):
        predicted = [10, 10, 20, 20]
        groundtruth = [10, 10, 20, 20]
        iou = calculate_iou(predicted, groundtruth)

        self.assertAlmostEqual(iou, 1.0)

    def test_half_overlap(self):
        predicted = [10, 10, 19, 20]
        groundtruth = [10, 10, 29, 20]
        iou = calculate_iou(predicted, groundtruth)

        self.assertAlmostEqual(iou, 0.5)

    def test_no_overlap(self):
        predicted = [10, 10, 20, 20]
        groundtruth = [25, 25, 30, 30]
        iou = calculate_iou(predicted, groundtruth)

        self.assertAlmostEqual(iou, 0.0)


if __name__ == '__main__':
    unittest.main()
