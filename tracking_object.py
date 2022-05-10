
from matplotlib.transforms import Bbox


class TrackingObject:
    def __init__(self, bbox, points):
        self.bbox = bbox
        self.points = points

    def update_bbox(self, new_points, status, depth_frame, cam):
        avg_x = 0
        avg_y = 0
        discarded_points = []
        # avg_depth = self.compute_avg_depth(depth_frame, cam)
        (xmin, ymin), (xmax, ymax) = self.bbox
        center_x = (xmax - xmin)/2 + xmin
        center_y = (ymax - ymin)/2 + ymin

        d = depth_frame[int(center_y), int(
            center_x)].astype(float)
        avg_depth = d * cam.depth_scale

        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = self.points[status == 1]
        else:
            return None

        good_points = 0
        for (new, old) in zip(good_new, good_old):
            z = 0
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            diff_x = x_new - x_old
            diff_y = y_new - y_old

            if y_old < cam.height and x_old < cam.width:
                depth = depth_frame[int(y_old), int(
                    x_old)].astype(float)
                z = depth * cam.depth_scale
                # print(f'Depth is {z}, average depth is {avg_depth}')
                # print(f'Deviation is {z/avg_depth}')

            if z >= 1.20 * avg_depth or z <= 0.80 * avg_depth:
                # print('point discarded------')
                discarded_points.append([int(x_old), int(y_old)])
            elif avg_depth == 0 or z == 0:
                avg_x += diff_x
                avg_y += diff_y
                good_points += 1
            else:
                avg_x += diff_x
                avg_y += diff_y
                good_points += 1

        if good_points != 0:
            avg_x = int(avg_x / good_points)
            avg_y = int(avg_y / good_points)
        elif len(good_old) != 0 and len(good_new) != 0:
            avg_x = int(avg_x / len(good_old))
            avg_y = int(avg_y / len(good_old))

        new_bbox = [(xmin + avg_x, ymin +
                    avg_y), (xmax + avg_x, ymax + avg_y)]
        self.bbox = new_bbox
        self.points = new_points
        return self.bbox, discarded_points

    def compute_avg_depth(self, depth_frame, cam):
        avg = 0
        for pt in self.points:
            x, y = pt.ravel()
            if y < cam.height and x < cam.width:
                depth = depth_frame[int(y), int(
                    x)].astype(float)
                z = depth * cam.depth_scale
                avg += z
        avg /= len(self.points)
        return avg
