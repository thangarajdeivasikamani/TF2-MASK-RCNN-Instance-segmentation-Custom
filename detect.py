import imghdr
import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import pathlib
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from utils.utils import encodeImageIntoBase64

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


class Predictor:
    def __init__(self):
        self.model = tf.saved_model.load("my_model\saved_model")
        self.category_index = label_map_util.create_category_index_from_labelmap("labelmap.pbtxt",
                                                                                 use_display_name=True)

    def load_image_into_numpy_array(self, path):
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    

    def run_inference(self):
        image_path = "inputImage.jpg"
        image_np = self.load_image_into_numpy_array(image_path)
        #print(image_np.shape)
        # Actual detection.
        model = self.model
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = model(input_tensor)
        #print(detections)
        # detection_classes should be ints.
        image_np_with_detections = image_np.copy()
        category_index = self.category_index
        # The following processing is only for single image
        detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
        detection_masks = tf.squeeze(detections['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0],
                                [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

        detections['num_detections'] = int(detections['num_detections'][0])
        detections['detection_classes'] = detections['detection_classes'][0].numpy(
        ).astype(np.uint8)
        detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
        detections['detection_scores'] = detections['detection_scores'][0].numpy()
        detections['detection_masks'] = detections['detection_masks'][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            instance_masks=detections.get('detection_masks'),
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            line_thickness=3)

        output_filename = 'output.jpg'
        cv2.imwrite(output_filename, image_np_with_detections)
        opencodedbase64 = encodeImageIntoBase64("output.jpg")
        #listOfOutput = []
        result = {"image": opencodedbase64.decode('utf-8')}
        return result

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
#     parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
#     parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
#     parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
#     args = parser.parse_args()

# detection_model = load_model(args.model)
# category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
#
# run_inference(detection_model, category_index, args.image_path)


