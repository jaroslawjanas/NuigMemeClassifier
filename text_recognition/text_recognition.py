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

    if mean > 184:
        _, processed_image = cv2.threshold(processed_image, 190, 255, cv2.THRESH_BINARY)
    else:
        _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 3)

    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_area = width*height
    print("  Text image area: ",image_area)
    # use weaker dilation for small text images
    if image_area < 20000 or height < 100:
        print("  Image size: Small")
        processed_image = cv2.dilate(processed_image, (3, 3), iterations=3)
    # use medium dilation for medium sized text images
    elif image_area < 40000 or height < 170:
        print("  Image size: Medium")
        processed_image = cv2.dilate(processed_image, (5, 5), iterations=3)
    # use strong dilation for big images
    else:
        print("  Image size: Large")
        processed_image = cv2.dilate(processed_image, (9, 9), iterations=5)


    cv2.imshow("Text image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return processed_image