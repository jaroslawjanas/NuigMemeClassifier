from text_detection.text_detection import *
from text_recognition.text_recognition import *
import yaml

config = ""
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# img_path = '../9gagScrapperJS_v3.0/data/part_19-1-2021_23h8m/images/a1rZQxG.jpg'
for i in range(1, 17):
    img_path = 'test_images/test' + str(i) + '.jpg'
    raw_image = cv2.imread(img_path)
    bounding_boxes = text_detection(raw_image)
    text_recognition(bounding_boxes, raw_image, config)


