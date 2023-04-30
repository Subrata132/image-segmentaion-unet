import numpy as np


class PixelFixer:
    def __init__(self):
        self.pixel_class = {
            'void': [(0, 0, 0)],
            'ground': [(128, 64, 128), (81, 0, 81)],
            'construction': [(230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153)],
            'object': [(153, 153, 153)],
            'nature': [(107, 142, 35), (152, 251, 152)],
            'sky': [(70, 130, 180)],
            'human': [(220, 20, 60), (255, 0, 0)],
            'vehicle': [(0, 0, 142), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110),
                        (0, 80, 110), (0, 0, 230), (119, 11, 32)]
        }
        self.class_to_ids = {class_: i for i, class_ in enumerate(self.pixel_class.keys())}

    def find_nearest(self, rgb):
        r, g, b = rgb
        distances = []
        for key, values in self.pixel_class.items():
            for value in values:
                R, G, B = value
                distance = ((R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2) ** .5
                distances.append((key, value, distance))
        distances = sorted(distances, key=lambda x: x[2])
        nearest_pixel, class_ = distances[0][1], distances[0][0]
        return nearest_pixel, class_

    def fix(self, label):
        # label_transposed = np.transpose(label, [2, 0, 1])
        corrected_label = np.zeros((3, 256, 256), dtype='uint8')
        classes = []
        for i in range(label.shape[1]):
            for j in range(label.shape[2]):
                rgb = label[:, i, j]
                neatest_pixel, class_ = self.find_nearest(rgb)
                classes.append(self.class_to_ids[class_])
                corrected_label[:, i, j] = neatest_pixel
        corrected_label = np.transpose(corrected_label, [1, 2, 0])
        return corrected_label, classes

