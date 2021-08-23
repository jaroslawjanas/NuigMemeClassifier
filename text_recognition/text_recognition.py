import cv2
import pytesseract

def text_recognition(bounding_boxes, raw_image, config):
    # path to tesseract-ocr
    pytesseract.pytesseract.tesseract_cmd = config['tesseract']['path']

    for box in bounding_boxes:
        print("----------------------")
        p1 = box[0]
        p3 = box[2]
        # copy part of image with text
        text_image = raw_image[int(p1[1]):int(p3[1]), int(p1[0]):int(p3[0])]

        cv2.imshow("Text image", text_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        processed_image = text_image_processing(text_image)
        text = pytesseract.image_to_string(processed_image, 'eng', '')
        print("  Text: \n", text)

    return None


def text_image_processing(text_image):
    height, width = text_image.shape[:2]
    print("  Text image dimensions: ", width, "x", height)

    processed_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(processed_image)[0]
    print("  Text image mean: ", mean)

    if width > 20 and height > 20:
        processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)

    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # if black text on white background - easy to threshold
    if mean > 180:
        _, processed_image = cv2.threshold(processed_image, 190, 255, cv2.THRESH_BINARY)
    # if white text on mixed background - hard to find threshold for
    else:
        counter = 0
        threshold_value = 200
        while True:
            counter += 1
            _, temp_processed_image = cv2.threshold(processed_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
            mean = cv2.mean(temp_processed_image)[0]
            print("  Mean after ths: ", mean)
            print("  Threshold value: ", threshold_value)
            # _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, -10)

            if counter > 20:
                _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                break
            elif mean < 50:
                threshold_value += 5
            elif mean > 200:
                threshold_value -= 5
            else:
                processed_image = temp_processed_image
                break

        cv2.imshow("Text image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # if image was hard to find threshold for don't dilate
        # as it's most likely of poor quality
        if counter < 3:
            processed_image = cv2.dilate(processed_image, (3, 3), iterations=1)

    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return processed_image