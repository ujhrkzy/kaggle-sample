# -*- coding: utf-8 -*-
import gc
from collections import defaultdict
from scipy.stats import itemfreq
from PIL import Image as IMG
import numpy as np
import pandas as pd
import operator
import cv2
import os
from kaggle_logger import logger


__author__ = "ujihirokazuya"

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

train_image_directory = "/home/res/data/competition_files/train_jpg"
test_image_directory = "/home/res/data/competition_files/test_jpg"
data_directory = "/home/res/"


class ImageFeatures(object):

    def __init__(self):
        self.dullness = None
        self.whiteness = None
        self.average_pixel_width = None

        self.dominant_red = None
        self.dominant_green = None
        self.dominant_blue = None

        self.average_red = None
        self.average_green = None
        self.average_blue = None

        self.image_size = None
        self.width = None
        self.height = None
        self.blurriness = None


class ImageFeatureExtractor(object):

    def __init__(self, image_name: str, image_directory: str):
        if image_name is None or not isinstance(image_name, str) or len(image_name) == 0:
            self._features = None
            return
        self._features = ImageFeatures()
        image_name = image_name + ".jpg"
        self._image_name = image_name
        path = os.path.join(image_directory, image_name)
        self._path = path
        # TODO confirm to need converting("RGB")
        self._pillow_image = IMG.open(path)
        self._cv_image = cv2.imread(path)

    def extract(self) -> ImageFeatures:
        if self._features is None:
            return ImageFeatures()
        dullness, whiteness = self._get_brightness()
        features = self._features
        features.dullness = dullness
        features.whiteness = whiteness
        features.average_pixel_width = self._get_average_pixel_width()

        dominant_color = self._get_dominant_color()
        features.dominant_red = dominant_color[0] / 255
        features.dominant_green = dominant_color[1] / 255
        features.dominant_blue = dominant_color[2] / 255

        average_color = self._get_average_color()
        features.average_red = average_color[0] / 255
        features.average_green = average_color[1] / 255
        features.average_blue = average_color[2] / 255

        features.image_size = self._get_size()
        dimensions = self._get_dimensions()
        features.width = dimensions[0]
        features.height = dimensions[1]

        features.blurriness = self._get_blurriness_score()
        return features

    def _get_brightness(self):
        # cut the images into two halves as complete average may give bias results
        size = self._pillow_image.size
        halves = (size[0] / 2, size[1] / 2)
        im1 = self._pillow_image.crop((0, 0, size[0], halves[1]))
        im2 = self._pillow_image.crop((0, halves[1], size[0], size[1]))

        def _analyse_color(img):
            # obtain the color pallet of the image
            pallet = defaultdict(int)
            for pixel in img.getdata():
                pallet[pixel] += 1

            # sort the colors present in the image
            sorted_x = sorted(pallet.items(), key=operator.itemgetter(1), reverse=True)
            light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
            for i, x in enumerate(sorted_x[:pixel_limit]):
                if all(xx <= 20 for xx in x[0][:3]): # dull : too much darkness
                    dark_shade += x[1]
                if all(xx >= 240 for xx in x[0][:3]): # bright : too much whiteness
                    light_shade += x[1]
                shade_count += x[1]
            light_percent = round((float(light_shade)/shade_count)*100, 2)
            dark_percent = round((float(dark_shade)/shade_count)*100, 2)
            return light_percent, dark_percent

        light_percent1, dark_percent1 = _analyse_color(im1)
        light_percent2, dark_percent2 = _analyse_color(im2)

        light_percent = (light_percent1 + light_percent2) / 2
        dark_percent = (dark_percent1 + dark_percent2) / 2
        return light_percent, dark_percent

    def _get_average_pixel_width(self):
        # im_array = np.asarray(self._pillow_image.convert(mode='L'))
        # edges_sigma1 = feature.canny(im_array, sigma=3)
        edges_sigma1 = cv2.Canny(self._cv_image, 100, 200)
        apw = (float(np.sum(edges_sigma1)) / (self._cv_image.shape[0] * self._cv_image.shape[1]))
        return apw * 100

    def _get_dominant_color(self):
        arr = np.float32(self._cv_image)
        pixels = arr.reshape((-1, 3))

        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        palette = np.uint8(centroids)
        dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
        return dominant_color

    def _get_average_color(self):
        average_color = [self._cv_image[:, :, i].mean() for i in range(self._cv_image.shape[-1])]
        return average_color

    def _get_size(self):
        st = os.stat(self._path)
        return st.st_size

    def _get_dimensions(self):
        return self._pillow_image.size

    def _get_blurriness_score(self):
        image = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(image, cv2.CV_64F).var()
        return fm


class ImagePreprocessor(object):

    def __init__(self):
        pass

    def execute(self):
        train_df = pd.read_csv(data_directory + "train.csv")
        test_df = pd.read_csv(data_directory + "test.csv")
        # train_df = pd.read_csv("top1000_train.csv")
        # test_df = pd.read_csv("top1000_test.csv")

        train_df = train_df.drop("deal_probability", axis=1)

        valid_columns = ["item_id", "image"]
        invalid_columns = [name for name in test_df.columns.values if name not in valid_columns]
        train_df = train_df.drop(invalid_columns, axis=1)
        test_df = test_df.drop(invalid_columns, axis=1)
        gc.collect()

        def _extract_train_feature(image_name: str) -> ImageFeatures:
            extractor = ImageFeatureExtractor(image_name, train_image_directory)
            return extractor.extract()

        train_df["features"] = train_df["image"].apply(_extract_train_feature)
        train_df = self._convert(train_df)
        train_df.to_csv("preprocessed_image_train.csv", index=False)
        del train_df
        gc.collect()

        def _extract_test_feature(image_name: str) -> ImageFeatures:
            extractor = ImageFeatureExtractor(image_name, test_image_directory)
            return extractor.extract()

        test_df["features"] = test_df["image"].apply(_extract_test_feature)
        test_df = self._convert(test_df)
        test_df.to_csv("preprocessed_image_test.csv", index=False)

    @staticmethod
    def _convert(data_frame):
        data_frame["dullness"] = data_frame["features"].apply(lambda x: x.dullness)
        data_frame["whiteness"] = data_frame["features"].apply(lambda x: x.whiteness)
        data_frame["average_pixel_width"] = data_frame["features"].apply(lambda x: x.average_pixel_width)

        data_frame["dominant_red"] = data_frame["features"].apply(lambda x: x.dominant_red)
        data_frame["dominant_green"] = data_frame["features"].apply(lambda x: x.dominant_green)
        data_frame["dominant_blue"] = data_frame["features"].apply(lambda x: x.dominant_blue)

        data_frame["average_red"] = data_frame["features"].apply(lambda x: x.average_red)
        data_frame["average_green"] = data_frame["features"].apply(lambda x: x.average_green)
        data_frame["average_blue"] = data_frame["features"].apply(lambda x: x.average_blue)

        data_frame["image_size"] = data_frame["features"].apply(lambda x: x.image_size)
        data_frame["width"] = data_frame["features"].apply(lambda x: x.width)
        data_frame["height"] = data_frame["features"].apply(lambda x: x.height)
        data_frame["blurriness"] = data_frame["features"].apply(lambda x: x.blurriness)
        data_frame = data_frame.drop("features", axis=1)
        return data_frame


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>> >>>>>>>>")
    try:
        pre_processor = ImagePreprocessor()
        pre_processor.execute()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>")
