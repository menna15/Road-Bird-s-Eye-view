import numpy as np
import cv2
import imutils
circlesPos=np.zeros((4,2),np.int) #4 position for the corners{ 2D }
index=0
def getMouseClick(event,x,y,flags,params): #fun to define Square Corners by Mouse
    global index
    if event == cv2.EVENT_LBUTTONDOWN and not index==4:
        circlesPos[index]= x,y
        print(circlesPos[index])
        createCircles(index)
        index=index+1

def createCircles(i): #create circle on the clicked point
    cv2.circle(image, (circlesPos[i][0], circlesPos[i][1]), 5, (0, 256, 450), cv2.FILLED)
def Prespective(): # will be called just in case we used MouseClick to define Square Corners
 while True:
  if index==4:
   cv2.imshow("Original", image)

   pts1=np.float32([circlesPos[0],circlesPos[1],circlesPos[2],circlesPos[3]])
   pts2 = np.float32([[250, 400], [325, 400], [325, 475], [250, 475]])
   matrix=cv2.getPerspectiveTransform(pts1,pts2)
   result=cv2.warpPerspective(image,matrix,(930,600))
   cv2.imwrite('birdsView_version2.jpg',result)
   cv2.imshow("Bird's Eye View", result)
   break

  cv2.imshow("Original", image)
  cv2.waitKey(1)
  cv2.setMouseCallback("Original", getMouseClick)
#***************** Main() **************************************
image = cv2.imread('road_titled.jpg')
image_cpy= cv2.imread('road_titled.jpg')
Original= cv2.imread('road_titled.jpg',0)
Prespective() #if you want to select the corners of the square manually

#  Details #
k=cv2.waitKey()
if k=='s':
    cv2.destroyAllWindows()
