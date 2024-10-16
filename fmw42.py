import cv2
import numpy as np
import os

image_path = 'image-samples/input/sample/page_1_gray.png'
print("Loading image from:", image_path)

if not os.path.isfile(image_path):
    print("Image file not found:", image_path)
else:
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
    else:
        print("Image loaded successfully:", image.shape)

        # get grayscale and binary image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # apply morphology close
        kernel = np.ones((35, 200), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # show grayscale and binary image and morphology image
        cv2.imshow('Grayscale Image', gray)
        cv2.imshow('Binary Image', binary)
        cv2.imshow('Morphological Closed Image', morph)

        # get contours and sort them
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        i = 1
        bboxes = image.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > 75 and w > 800:
                # draw contour on input
                # cv2.drawContours(contour_img, [c], 0, (0,0,255), 1)

                # draw bounding boxes on input
                cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # crop image for given bounding box and save
                crop = image[y:y + h, x:x + w]
                cv2.imwrite("page_2_gray_crop{0}.png".format(i), crop)
                i = i + 1

        # show bounding boxes
        cv2.imshow('Bounding Boxes', bboxes)
        cv2.waitKey(0)

        # save images
        cv2.imwrite('page_2_gray_gray.png', gray)
        cv2.imwrite('page_2_gray_binary.png', binary)
        cv2.imwrite('page_2_gray_morph.png', morph)
        cv2.imwrite('output-issues/page_1_gray_bounding_boxes.png', bboxes)

        print("Number of contours detected:", len(contours))