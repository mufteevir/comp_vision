"""
Let's try to simply use a threshold and then use findContours.
result is only external contour
"""
import cv2
import matplotlib.pyplot as plt


def display(img, cmap=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img, cmap=cmap)
    plt.show()


sep_coins = cv2.imread('DATA/pennies.jpg')

#Apply Median Blurring
#We have too much detail in this image, including light,
# the face edges on the coins, and too much detail in the background.
# Let's use Median Blur Filtering to blur the image a bit,
# which will be useful later on when we threshold.

sep_blur = cv2.medianBlur(sep_coins,25)
gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)

#Binary Threshold

ret, sep_thresh = cv2.threshold(gray_sep_coins,160,255,cv2.THRESH_BINARY_INV)
display(sep_thresh,cmap='gray')

#FindContours

contours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# For every entry in contours
for i in range(len(contours)):

    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        # We can now draw the external contours from the list of contours
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)

display(sep_coins)

