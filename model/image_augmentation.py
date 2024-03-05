import random
import cv2
import numpy as np

class Augmentation():
    def random_salt_and_papper(image):
        #salt and papper
        width = image.shape[1]
        height = image.shape[0]
        
        salts = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

        for salt in salts:
            if random.random() < 0.1:
                salt_count = int(width * height * 0.01 * random.random())
                y_coord = np.random.randint(0, height, [salt_count])
                x_coord = np.random.randint(0, width, [salt_count])
                image[(y_coord, x_coord)] = salt


    def random_blur_or_shappen(image):
        blured = cv2.GaussianBlur(image, (5, 5), 7)
        weight = np.array(-0.5 + random.random() * 0.7, dtype=np.float32)
        image = image * (1 - weight) + blured * weight
        return image  


    def random_flip(image):
        if random.random() > 0.5:
            image =  cv2.flip(image, 1)
        return image


    def random_dim(image):
        for ch in range(0, 3):
    #         image[:, :, ch] += random.randint(0, 10)
            image[:, :, ch] += 40
        return image

    def random_ditter(image):
        if random.random() < 0.5:
            image += np.random.normal(0, random.randint(1, 20), image.shape)


    def random_multi_tone(image):
        if random.random() < 0.1:
            sep = random.randint(image.shape[1] // 4, image.shape[1] * 3 // 4)
            offset = (np.random.random([3]) - 0.5) * 30
            image[:, :sep, :] += offset
            offset = (np.random.random([3]) - 0.5) * 30
            image[:, sep:, :] += offset


    def random_box(image):
        for _ in range(3):
            if random.random() < 0.3:
                bw = random.randint(10, image.shape[1] // 5)
                bh = random.randint(10, image.shape[0] // 5)
                bx = random.randint(0, image.shape[1] - bw)
                by = random.randint(0, image.shape[0] - bh)
                color = np.random.randint(0, 255, (3,))
                image[by:by+bh, bx:bx+bw, :] = color

    def random_shadow(image):
        width = image.shape[1]
        height = image.shape[0]

        # ver_shadow = (np.arange(height) - (height / 2)) / (height / random.randint(1, 30))
        # hor_shadow = (np.arange(width) - (width / 2)) / (width / random.randint(1, 30))

        ver_shadow = np.linspace(0, 1, height).reshape([-1, 1, 1])
        hor_shadow = np.linspace(0, 1, width).reshape([1, -1, 1])
        
        ver_shadow *= np.random.randn() * 40
        hor_shadow *= np.random.randn() * 40

        image += ver_shadow.astype(np.float32)
        image += hor_shadow.astype(np.float32)