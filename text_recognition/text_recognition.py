import cv2
import pytesseract

def text_recognition(bounding_boxes, raw_image, config):
    # path to tesseract-ocr
    pytesseract.pytesseract.tesseract_cmd = config['tesseract']['path']

    for box in bounding_boxes:
        p1 = box[0]
        p3 = box[2]
        # copy part of image with text
        text_image = raw_image[int(p1[1]):int(p3[1]), int(p1[0]):int(p3[0])]

        cv2.imshow("Text image", text_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        processed_image = text_image_processing(text_image)
        text = pytesseract.image_to_string(processed_image, 'eng', '')
        print(text)

    return None


def text_image_processing(text_image):
    height, width = text_image.shape[:2]

    processed_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    if width > 20 and height > 20:
        processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)

    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    # _, processed_image = cv2.threshold(processed_image, 230, 255, cv2.THRESH_BINARY)

    # processed_image = cv2.dilate(processed_image, (3, 3), iterations=1)
    # processed_image = cv2.erode(processed_image, (3, 3), iterations=2)
    #
    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return processed_image