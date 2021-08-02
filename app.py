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

for i in range(11, 17):
    input_img_path = 'test_images/test' + str(i) + '.jpg'
    input_img = cv2.imread(input_img_path)
    bounding_boxes = text_detection(input_img, config)
    text_recognition(bounding_boxes, input_img, config)



    templates = os.listdir(config['data']['templates']['path'])
    templatePaths = [config['data']['templates']['path'] + p + '\\' + p+ '.jpg' for p in templates]

    image_matching(input_img, bounding_boxes, templatePaths)

