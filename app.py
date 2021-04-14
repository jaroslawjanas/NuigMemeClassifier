from text_detection.text_detection import *

# img_path = '../9gagScrapperJS_v3.0/data/part_19-1-2021_23h8m/images/a1rZQxG.jpg'
for i in range(1, 17):
    img_path = 'test_images/test' + str(i) + '.jpg'
    img = image_processing(img_path)

