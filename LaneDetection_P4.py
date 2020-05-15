# THIS FILE IS PART OF UE4 LIDAR SIMULATE AND AUTO DRIVE SIMULATE PROJECT 
# THIS PROGRAM IS FREE SOFTWARE, IS LICENSED UNDER MIT
# Copyright(c) Bowei Zhang
# https://github.com/GGzhangBOY/AlgContainer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import DataReceiver
import cv2
import math
import Behaviors

haveShown = False
Tracked = False
imageShape = []
sim_car = Behaviors.SimCar()
g_left_fit_x = []
g_lane_center = 0

def calcaulateOffsetAndAngle(curve_x_l, lane_center):
    l_curve_vector = [curve_x_l[int(len(curve_x_l)/2)] - curve_x_l[0],int(len(curve_x_l)/2)]

    result = [math.degrees(math.atan2(l_curve_vector[0],l_curve_vector[1])), imageShape[1]/2 - lane_center]

    print(result)
    return result


def show_info(img,left_cur,right_cur,center):

    cur = (left_cur + right_cur) / 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img,'Curvature = %d(m)' % cur,(50,50),font,1,(255,255,255),2)

    
    if center < 0:
        fangxiang = 'left'
    else:
        fangxiang = 'right'
        
    cv2.putText(img,'the angle is %.2fm of %s'%(np.abs(center),fangxiang),(50,100),font,1,(255,255,255),2)

    cv2.putText(img,'State: Tracked',(50,150),font,1,(255,255,255),2)

def draw_lines(undist,warped,left_fit,right_fit,left_cur,right_cur,center,Minv,show_img = True):

    undist = np.array(undist)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    
    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])

    left_fitx = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]
    

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 

    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    show_info(result, left_cur, right_cur, center)
    if show_img == True:
        plt.figure(figsize = (10,10))
        plt.imshow(result)
        plt.show()
    return result

def curvature(left_fit,right_fit,binary_warped,print_data = True):
    ploty = np.linspace(0,binary_warped.shape[0] -1 , binary_warped.shape[0])
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    

    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    

    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix
    
    if print_data == True:

        print(left_curverad, 'm', right_curverad, 'm', center, 'm')
 
    return left_curverad, right_curverad, center

def find_lines(img,print = True):
    global Tracked

    histogram= np.sum(img[img.shape[0] //2:,:],axis = 0)

    out_img = np.dstack((img,img,img))*255

    midpoint = np.int(histogram.shape[0] // 4)
    leftx_base = np.argmax(histogram[:midpoint])

    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    nwindows = 9

    window_height = np.int(img.shape[0] // nwindows)

    nonzero = img.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100

    minpix = 50

    left_lane_inds = []
    right_lane_inds = []
 
    for window in range(nwindows):

        win_y_low = img.shape[0] - (window + 1) * window_height 

        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin

        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin

        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
 
        good_left_inds = (  (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high)  
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
 
 
        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                              & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
 

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    

    left_lane_inds = np.concatenate(left_lane_inds)

    right_lane_inds = np.concatenate(right_lane_inds)
    

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    if(len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0):
        Tracked = False
        return [0,0,0]
    else:
        Tracked = True


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

    ploty = np.linspace(0,img.shape[0] -1,img.shape[0]) 
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]

    
    lane_center = (left_fit[2] + right_fit[2]) / 2
    
    global g_left_fit_x 
    global g_lane_center 

    g_left_fit_x = left_fitx
    g_lane_center = lane_center
    #calcaulateOffsetAndAngle(left_fitx, lane_center)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if print == True:
        plt.figure(figsize=(8,8))
        
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.show()
    
    return out_img,left_fit,right_fit

def warp(edged_img, from_poly, to_poly, print_img = True):

    img_size = (edged_img.shape[1],edged_img.shape[0])
    M = cv2.getPerspectiveTransform(np.float32(from_poly),np.float32(to_poly))
    Minv = cv2.getPerspectiveTransform(np.float32(to_poly),np.float32(from_poly))
    
    warped = cv2.warpPerspective(edged_img,M,img_size,flags = cv2.INTER_LINEAR)
    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)

    if print_img:
        cv2.imshow("Edges", warped)
        cv2.waitKey(1)

    return warped, unpersp, Minv

def AlgOutput(cv_result):
    if(cv_result[1] > 40):
        return 5
    elif(cv_result[1] < -20):
        return -5
    else:
        return cv_result[0]


while True:

    _,image_org = DataReceiver.aquireImageData(True)
    gray = image_org[2]

    image_dim3 = np.dstack((image_org[2], image_org[2], image_org[2]))

    kernel_size = 17
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)


    low_threshold = 30
    high_threshold = 35
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    imshape = gray.shape
    imageShape = gray.shape

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

    warped, unpersp, Minv = warp(edges, ROI_Arear, ToTransform_Arear, False)
    out_img, left_fit, right_fit = find_lines(warped , False)

    if(Tracked):
        left_curverad, right_curverad, center = curvature(left_fit,right_fit,warped, False)
        result = draw_lines(image_dim3,warped,left_fit,right_fit,left_curverad,right_curverad,center,Minv, False)
        to_adjust = calcaulateOffsetAndAngle(g_left_fit_x, g_lane_center)
        Alg_result = AlgOutput(to_adjust)
        sim_car.GetAlgInputAndWrite(-Alg_result, 0, 0.5)
    else:
        result = gray
        sim_car.GetAlgInputAndWrite(0, 1, 0)
        
    cv2.imshow("Camera3",cv2.resize(result, (640,480), interpolation=cv2.INTER_AREA))
    cv2.waitKey(1)    