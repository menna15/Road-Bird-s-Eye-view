import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as matImg
def Get_squareDetection_by_Contour(img):
    GrayImg = cv2.bilateralFilter(img, 11, 17, 18)
    Edged = cv2.Canny(GrayImg, 30, 200)
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    Contors = cv2.findContours(Edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contors = imutils.grab_contours(Contors)
    Contors = sorted(Contors, key=cv2.contourArea, reverse=True)[:10]

    SquareContour = None
    # loop over our contours
    for c in Contors:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen

        if len(approx) == 4:
            SquareContour = approx
            x_y_array = approx.ravel()
            break
    return SquareContour, x_y_array

def DrawContor(screenCnt,n):
    cv2.drawContours(image, [screenCnt], -1, (200,0,255), 4)
    Draw_coordinates(image,n)
    birdsView(screenCnt,n)
    cv2.imshow("Original", image)

def Draw_coordinates(img,Contors_list): #Write CoOrdinates of the Square
     i = 0
     while i <7:
      cv2.putText(img,str((Contors_list[i], Contors_list[i+1])), (Contors_list[i], Contors_list[i+1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,0,255),1)
      i=i+2


def birdsView(screenCnt,n): #Creates Top View image
    global square_width
    pts1 = np.array(screenCnt, dtype="float32")
    pts2 = np.float32([[250, 400], [325, 400], [325, 475], [250, 475]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    result=cv2.warpPerspective(image_cpy,matrix,(930,600))
    cv2.imwrite('birdsView.jpg',result)
    result_cpy=cv2.imread('birdsView.jpg')
    s,f=Get_squareDetection_by_Contour(result_cpy)
    square_width=abs(f[0]-f[1])
    Draw_coordinates(result_cpy,f)
    cv2.imshow("Bird's Eye View", result_cpy)


#//////////////////////////// For Lane Detection ///////////////////////////////////////
region_of_interest_vertices = [
    (135, 0),
    (215, 520),
    (555, 0),
    (500,578)
]
def Region_of_Interest(img, vertices):
    Mask = np.zeros_like(img)
    match_mask_color = 255  # <-- This line altered for grayscale.

    cv2.fillPoly(Mask, vertices, match_mask_color)
    Masked_img = cv2.bitwise_and(img, Mask)
    return Masked_img

def Draw_lines(img, lines, color=[0, 0, 257], thickness=4):
    # If there are no lines to draw, exit.
    if lines is None:
            return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def LaneDetection(): #main func of laneDetection part
 global Road_width
 source_img = matImg.imread('birdsView.jpg')
 Gray_Img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
 Cannyed_image = cv2.Canny(Gray_Img, 350, 300)
 Cropped_image = Region_of_Interest(
    Cannyed_image,
    np.array(
        [region_of_interest_vertices],
        np.int32
    ),
 )
 Lines = cv2.HoughLinesP(
    Cropped_image,
    rho=8,
    theta=np.pi/20,
    threshold=140,
    lines=np.array([]),
    minLineLength=70,
    maxLineGap=50
)
 #print(Lines)
 Road_width=abs(Lines[0][0][0]-Lines[1][0][0])/100 # extract Road Width from 2 points on the Detected lines
 line_image = Draw_lines(source_img, Lines)
 cv2.putText(line_image,'__'+str(abs(Lines[0][0][0]-Lines[1][0][0])/100)+'meters__', (Lines[0][0][0]+10, Lines[1][0][0]-100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
 plt.figure()
 plt.imshow(line_image)
 plt.show()
#/////////////////////////////////////////////// lane detection part finished /////////////////////////////////////////
#***************** Main() **************************************
image = cv2.imread('road_titled.jpg')
image_cpy= cv2.imread('road_titled.jpg')
Original= cv2.imread('road_titled.jpg',0)

s,n=Get_squareDetection_by_Contour(Original)
DrawContor(s,n)
LaneDetection()
#  Details #
print("True Square Width : 75 cm")
print("Square Width in bird's Eye View :",str(square_width)+'cm')
print("True Road Width : ",str((75*Road_width/square_width))+'m')
print("Road Width in bird's Eye View :",str(Road_width)+'m')
k=cv2.waitKey()
if k=='s':
    cv2.destroyAllWindows()
