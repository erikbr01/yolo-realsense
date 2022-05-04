
from matplotlib.transforms import Bbox


class TrackingObject:
    def __init__(self, bbox, points):
        self.bbox = bbox
        self.points = points

    def update_bbox(self, new_points, status):
        avg_x = 0
        avg_y = 0

        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = self.points[status == 1]
        else:
            return None

        for (new, old) in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            diff_x = x_new - x_old
            diff_y = y_new - y_old

            avg_x += diff_x
            avg_y += diff_y

        avg_x = int(avg_x / len(good_new))
        avg_y = int(avg_y / len(good_new))
        (xmin, ymin), (xmax, ymax) = self.bbox
        new_bbox = [(xmin + avg_x, ymin +
                    avg_y), (xmax + avg_x, ymax + avg_y)]
        self.bbox = new_bbox
        self.points = new_points
        return self.bbox
