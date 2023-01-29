"""
Create a program that reads in a live stream from a camera on your computer
(or if you don't have a camera, just open up a video file).
Then whenever you click the left mouse button down,
create a blue circle around where you've clicked.
Check out the video for an example of what the final project should look like
"""
# Create a function based on a CV2 Event (Left button click)
import cv2

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global center, clicked

    # get mouse click on down and track center
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        clicked = False

    # Use boolean variable to track if the mouse has been released
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


# Haven't drawn anything yet!
center = (0, 0)
clicked = False

# Capture Video
cap = cv2.VideoCapture(0)

# Create a named window for connections
cv2.namedWindow('Test')

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback('Test', draw_circle)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Use if statement to see if clicked is true
    if clicked == True:
        # Draw circle on frame
        cv2.circle(frame, center=center, radius=50, color=(255, 0, 0), thickness=5)

    # Display the resulting frame
    cv2.imshow('Test', frame)

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
