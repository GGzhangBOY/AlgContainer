#doing all the relevant imports
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
    #在图片中显示出曲率
    cur = (left_cur + right_cur) / 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 使用默认字体
    cv2.putText(img,'Curvature = %d(m)' % cur,(50,50),font,1,(255,255,255),2)
    #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    #添加文字
    
    if center < 0:
        fangxiang = 'left'
    else:
        fangxiang = 'right'
        
    cv2.putText(img,'the angle is %.2fm of %s'%(np.abs(center),fangxiang),(50,100),font,1,(255,255,255),2)

    cv2.putText(img,'State: Tracked',(50,150),font,1,(255,255,255),2)

def draw_lines(undist,warped,left_fit,right_fit,left_cur,right_cur,center,Minv,show_img = True):
    #创建一个全黑的底层图去划线
    undist = np.array(undist)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    
    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
    #添加新的多项式在X轴Y轴
    left_fitx = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #把X和Y变成可用的形式
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    #np.transpose 转置
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #向上/向下翻转阵列。
    pts = np.hstack((pts_left, pts_right))
    #填充图像
    cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
    #透视变换
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    #叠加图层
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
    #y_eval就是曲率，这里是选择最大的曲率
    
    ym_per_pix = 30/720#在y维度上 米/像素
    xm_per_pix = 3.7/700#在x维度上 米/像素
    
    #确定左右车道
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #定义新的系数在米
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    #最小二乘法拟合
    
    #计算新的曲率半径
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #计算中心点，线的中点是左右线底部的中间
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix#转换成米
    
    if print_data == True:
        #现在的曲率半径已经转化为米了
        print(left_curverad, 'm', right_curverad, 'm', center, 'm')
 
    return left_curverad, right_curverad, center

def find_lines(img,print = True):
    global Tracked
    #假设您已经创建了一个被扭曲的二进制图像，称为“binary_warped”
    #取图像下半部分的直方图
    histogram= np.sum(img[img.shape[0] //2:,:],axis = 0)
    #创建一个输出图像来绘制和可视化结果
    out_img = np.dstack((img,img,img))*255
    # plt.imshow(out_img)
    # plt.show()
    #找出直方图的左半边和右半边的峰值
    #这些将是左行和右行的起点
    midpoint = np.int(histogram.shape[0] // 4)
    leftx_base = np.argmax(histogram[:midpoint])
    #np.argmax 是返回最大值所在的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #这里是要返回右边HOG值最大所在的位置，所以要加上midpoint
 
    #选择滑动窗口的数量
    nwindows = 9
    #设置窗口的高度
    window_height = np.int(img.shape[0] // nwindows)
    #确定所有的x和y位置非零像素在图像,这里就是吧img图像中非0元素（就是不是黑的地方就找出来，一行是x，一行是y）
    nonzero = img.nonzero()
    #返回numpy数组中非零的元素
    #对于二维数组b2，nonzero(b2)所得到的是一个长度为2的元组。http://www.cnblogs.com/1zhk/articles/4782812.html
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #为每个窗口当前位置更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    #设置窗口的宽度+ / -
    margin = 100
    #设置最小数量的像素发现重定位窗口
    minpix = 50
    #创建空的列表接收左和右车道像素指数
    left_lane_inds = []
    right_lane_inds = []
 
    #遍历窗口
    for window in range(nwindows):
        #识别窗口边界在x和y(左、右)
        win_y_low = img.shape[0] - (window + 1) * window_height #就是把图像切成9分，一分一分的算HOG
        #print('win_y_low',win_y_low)
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        #print('win_xleft_low',win_xleft_low)
        win_xleft_high = leftx_current + margin
        #print('win_xleft_high = ',win_xleft_high)
        win_xright_low = rightx_current - margin
        #print('win_xright_low = ',win_xright_low)
        win_xright_high = rightx_current + margin
        #print('win_xright_high = ',win_xright_high)
        #把网格画在可视化图像上
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)#通过确定对角线 画矩形
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
 
    #     plt.imshow(out_img)
    #     plt.show()
    #     print('left !!!! ',win_xleft_low,win_y_low,win_xleft_high,win_y_high)
    #     print('right !!!!! ',win_xright_low,win_y_low,win_xright_high,win_y_high)
 
        #识别非零像素窗口内的x和y
        good_left_inds = (  (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high)  
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
 
 
        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                              & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
 
        #添加这些指标列表
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #如果上面大于minpix，重新定位下一个窗口的平均位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #连接索引的数组
    left_lane_inds = np.concatenate(left_lane_inds)
    #把list改成numpy格式而已
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #提取左和右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    if(len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0):
        Tracked = False
        return [0,0,0]
    else:
        Tracked = True

    #最小二乘多项式拟合。(不懂)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #画图
    ploty = np.linspace(0,img.shape[0] -1,img.shape[0]) #用此来创建等差数列
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]
    #这步的意思是把曲线拟合出来，
    
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
    # Read in the image and convert to grayscale
    _,image_org = DataReceiver.aquireImageData(True)
    gray = image_org[2]

    image_dim3 = np.dstack((image_org[2], image_org[2], image_org[2]))
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