"""
Use the image under the DATA folder called car_plate.jpg and create a
function that will blur the image of its license plate.
Check out the Haar Cascades folder for the correct pre-trained .xml file to use
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read in the car_plate.jpg file from the DATA folder.
img = cv2.imread('DATA/car_plate.jpg')

def display(img):
    """
    Create a function that displays the image in a larger scale and correct coloring for matplotlib.
    :param img:
    :return:
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    plt.show()

# Load the haarcascade_russian_plate_number.xml file.
plate_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_russian_plate_number.xml')


def detect_plate(img):
    """
    Create a function that takes in an image and draws a rectangle around what it detects to
    be a license plate. Keep in mind we're just drawing a rectangle around it for now,
    later on we'll adjust this function to blur. You may want to play with the scaleFactor
    and minNeighbor numbers to get good results.
    :param img:
    :return:
    """
    plate_img = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 0, 255), 4)

    return plate_img

result = detect_plate(img)
display(result)

def detect_and_blur_plate(img):
    """
    Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:

    The hardest part is converting the (x,y,w,h) information into the dimension values you need to grab an ROI (somethign we covered in the lecture 01-Blending-and-Pasting-Images. It's simply Numpy Slicing, you just need to convert the information about the top left corner of the rectangle and width and height, into indexing position values.
    Once you've grabbed the ROI using the (x,y,w,h) values returned, you'll want to blur that ROI. You can use cv2.medianBlur for this.
    Now that you have a blurred version of the ROI (the license plate) you will want to paste this blurred image back on to the original image at the same original location. Simply using Numpy indexing and slicing to reassign that area of the original image to the blurred roi.
    :param img:
    :return:
    """

    plate_img = img.copy()
    roi = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in plate_rects:
        roi = roi[y:y + h, x:x + w]
        blurred_roi = cv2.medianBlur(roi, 7)

        plate_img[y:y + h, x:x + w] = blurred_roi

    return plate_img

result = detect_and_blur_plate(img)
display(result)