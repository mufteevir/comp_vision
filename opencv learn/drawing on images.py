#test program for learning drawing forms and text on picture
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("DATA/dog_backpack.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
# plt.imshow(img_rgb)
# plt.show()
#Flip the image upside down
new_img = img_rgb
new_img = cv2.flip(new_img,0)
#plt.imshow(new_img)
#Draw an empty RED rectangle around the dogs face
cv2.rectangle(img_rgb,pt1=(200,380),pt2=(600,700),color=(255,0,0),thickness=10)
# plt.imshow(img_rgb)
# plt.show()

#Draw a BLUE TRIANGLE in the middle of the image.
#To draw a polygon, first you need coordinates of vertices.
# Make those points into an array of shape ROWSx1x2 where
# ROWS are number of vertices and it should be of type int32.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vertices = np.array([[250,700],[425,400],[600,700]],np.int32)
pts = vertices.reshape((-1,1,2))

cv2.polylines(img_rgb,[pts],isClosed=True,color=(0,0,255),thickness=20)
# plt.imshow(img_rgb)
# plt.show()

#fill in this triangle
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vertices = np.array([[250,700],[425,400],[600,700]],np.int32)
pts = vertices.reshape((-1,1,2))

cv2.fillPoly(img_rgb,[pts],color=(0,0,255))
# plt.imshow(img_rgb)
# plt.show()

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img_rgb,pt1=(0,0),pt2=(511,511),color=(102, 255, 255),thickness=5)
# plt.imshow(img_rgb)
# plt.show()

#text

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_rgb,text='Hello',org=(10,500), fontFace=font,fontScale= 4,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
plt.imshow(img_rgb)
plt.show()


#Create a script that opens the picture and allows you to draw empty red circles
# whever you click the RIGHT MOUSE BUTTON DOWN.¶
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,0,255),10)

# Open Image
img = cv2.imread("DATA/dog_backpack.png")
# This names the window so we can reference it
cv2.namedWindow(winname='dog')
# Connects the mouse button to our callback function
cv2.setMouseCallback('dog',draw_circle)

while True: #Runs forever until we break with Esc key on keyboard
    # Shows the image window
    cv2.imshow('dog',img)
    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    if cv2.waitKey(20) & 0xFF == 27:
        break
# Once script is done, its usually good practice to call this line
# It closes all windows (just in case you have multiple windows called)
cv2.destroyAllWindows()

