from text_detection.text_detection import *
from text_recognition.text_recognition import *
import yaml
from image_classification.image_similarity import *
import os

config = ""
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

templates = os.listdir(config['data']['templates']['path'])
template_paths = [config['data']['templates']['path'] + p + '\\' for p in templates]

for i in range(1, 16):
    print('========================================================')
    input_img_path = 'test_images/test' + str(i) + '.jpg'
    input_img = cv2.imread(input_img_path)
    bounding_boxes = text_detection(input_img, config)
    text_recognition(bounding_boxes, input_img, config)
    # image_matching(input_img, bounding_boxes, templates, config)
    print('========================================================')
