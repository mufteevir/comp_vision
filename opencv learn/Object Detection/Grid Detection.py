import cv2
import matplotlib.pyplot as plt

flat_chess = cv2.imread('DATA/flat_chessboard.png')
plt.imshow(flat_chess,cmap='gray')
plt.show()

found, corners = cv2.findChessboardCorners(flat_chess,(7,7))
if found:
    print('OpenCV was able to find the corners')
else:
    print("OpenCV did not find corners. Double check your patternSize.")

flat_chess_copy = flat_chess.copy()
cv2.drawChessboardCorners(flat_chess_copy, (7, 7), corners, found)

plt.imshow(flat_chess_copy)
plt.show()

#Circle Based Grids

dots = cv2.imread('DATA/dot_grid.png')
plt.imshow(dots)
plt.show()

found, corners = cv2.findCirclesGrid(dots, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)
dbg_image_circles = dots.copy()
cv2.drawChessboardCorners(dbg_image_circles, (10, 10), corners, found)
plt.imshow(dbg_image_circles)
plt.show()