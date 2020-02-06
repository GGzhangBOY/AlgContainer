#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import DataReceiver
import cv2

haveShown = False
def warp(edged_img, from_poly, to_poly):

    img_size = (edged_img.shape[1],edged_img.shape[0])
    M = cv2.getPerspectiveTransform(np.float32(from_poly),np.float32(to_poly))
    Minv = cv2.getPerspectiveTransform(np.float32(to_poly),np.float32(from_poly))
    
    warped = cv2.warpPerspective(edged_img,M,img_size,flags = cv2.INTER_LINEAR)
    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)

    cv2.imshow("Edges", warped)
    cv2.waitKey(1)
    return warped, unpersp, Minv


while True:
    # Read in the image and convert to grayscale
    _,image_org = DataReceiver.aquireImageData(True)
    gray = image_org[2]

    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    kernel_size = 17
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Define parameters for Canny and run it
    # NOTE: if you try running this code you might want to change these!
    low_threshold = 30
    high_threshold = 35
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    imshape = gray.shape

    ROI_Arear = [(50,int(imshape[0]-50)),(int(imshape[1]/2-200),int(imshape[0]/2+100)), 
        (int(imshape[1]/2+200),int(imshape[0]/2+100)), (int(imshape[1]-50),int(imshape[0]-50))]
    ToTransform_Arear = [(50,int(imshape[0]-50)),(50,int(0)), 
        (imshape[1]-50,int(0)), (int(imshape[1]-50),int(imshape[0]-50))]

    """ROI_Arear = np.float32( [ [800,510],[1150,700],[270,700],[510,510]] )
    ToTransform_Arear = np.float32( [ [650,470],[640,700],[270,700],[270,540]] )"""
    if not haveShown:
        plt.imshow(edges,cmap = 'gray')
        plt.plot(50,int(imshape[0]-50),'o')
        plt.plot(int(imshape[1]/2-200),int(imshape[0]/2+100),'o')
        plt.plot(int(imshape[1]/2+200),int(imshape[0]/2+100),'o')
        plt.plot(int(imshape[1]-50),int(imshape[0]-50),'o')
        
        
        plt.plot(50,int(imshape[0]-50),'x')
        plt.plot(50,int(0),'x')
        plt.plot(imshape[1]-50,int(0),'x')
        plt.plot(int(imshape[1]-50),int(imshape[0]-50),'x')

        plt.show()
        haveShown = True

    warp(edges, ROI_Arear, ToTransform_Arear)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    
    vertices = np.array([[(50,imshape[0]-50),(imshape[1]/2-50,imshape[0]/2+25), 
        (imshape[1]/2+50,imshape[0]/2+25), (imshape[1]-50,imshape[0]-50)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150 #minimum number of pixels making up a line
    max_line_gap = 100   # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    line_image = np.dstack((line_image, line_image, line_image))
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    # Display the image
    cv2.imshow("Camera3",lines_edges)
    cv2.waitKey(1)