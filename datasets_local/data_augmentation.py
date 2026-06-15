import numpy as np

class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img_tensor):
        # img_tensor è un Tensor C×H×W già normalizzato o no
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        y_center = np.random.randint(h)
        x_center = np.random.randint(w)

        y1 = np.clip(y_center - self.length // 2, 0, h)
        y2 = np.clip(y_center + self.length // 2, 0, h)
        x1 = np.clip(x_center - self.length // 2, 0, w)
        x2 = np.clip(x_center + self.length // 2, 0, w)

        img_tensor[:, y1:y2, x1:x2] = 0.0
        return img_tensor
