import mmap
from ctypes import *
import binascii
import numpy as np
from PIL import Image
import PIL.Image as img
from cv2 import cv2 as cv

class Position(Structure):
    _fields_= [("x", c_float),("y", c_float),("z",c_float)]

class PC_info(Structure):
    _fields_= [("point_position",Position),("padding",c_int)]

class PointCloudPackage(Structure):
    _fields_= [("dataLength", c_int),("pointCloudInfo", POINTER(PC_info))]

class pixel_structure(Structure):
    _fields_= [("R",c_int8),("G",c_int8),("B",c_int8),("A",c_int8)]

class VideoStreamPack(Structure):
    _fields_= [("data", POINTER(POINTER(pixel_structure))),("height",c_int)
    ,("width", c_int), ("num_camera", c_int)]
Points = []

c_dll = cdll.LoadLibrary("SM_Access.dll")
g_carPos = c_dll.GetCarPosition
g_carPos.restype = Position
g_PCData = c_dll.GetPointCloudData
g_PCData.restype = PointCloudPackage
g_CameraData =c_dll.GetCarCameraData
g_CameraData.restype = VideoStreamPack
g_FreePointCloudData = c_dll.FreePointCloudMemary


hasShown = 0
while True:
    buff1 = g_carPos()
    print("Car position: ",buff1.x," ",buff1.y," ",buff1.z)

    PC_Origin = []
    for i in range(12):
        buff2 = g_PCData()
        

        PC_Origin = np.ctypeslib.as_array(cast(buff2.pointCloudInfo, POINTER(c_float))
        , (buff2.dataLength, 4))


        PC_Origin = np.delete(PC_Origin, -1, axis=1)

        """
        for j in range(buff2.dataLength):
            buff3 = buff2.pointCloudInfo[j]
            b_x = buff3.point_position.x
            b_y = buff3.point_position.y
            b_z = buff3.point_position.z
            Points.append([b_x,b_y,b_z])
            """
        g_FreePointCloudData()
        
        
    print("Points: ",buff2.dataLength)

    buff3 = g_CameraData()

    data = []
    img_bytes = []
    for i in range(buff3.num_camera):
        img_origin = np.ctypeslib.as_array(cast(buff3.data[i], POINTER(c_ubyte))
        , (buff3.height, buff3.width, 4))
        img_buf = np.delete(img_origin, -1, axis=1)
        img_bytes.append(img_buf)

    if(hasShown == 2):#change to 0 when debug
        t_image = Image.fromarray(img_bytes[0])
        Image._show(t_image.convert('L').transpose(img.FLIP_TOP_BOTTOM) )
        t_image1 = Image.fromarray(img_bytes[1])
        Image._show(t_image1.convert('L').transpose(img.FLIP_TOP_BOTTOM) )
        t_image2 = Image.fromarray(img_bytes[2])
        Image._show(t_image2.convert('L').transpose(img.FLIP_TOP_BOTTOM) )
        hasShown = 1
    
    t_image = Image.fromarray(img_bytes[0])
    cv.imshow("Camera1", cv.resize(np.array(t_image.convert('L').transpose(img.FLIP_TOP_BOTTOM)), (320,240), interpolation=cv.INTER_AREA))
    t_image1 = Image.fromarray(img_bytes[1])
    cv.imshow("Camera2", cv.resize(np.array(t_image1.convert('L').transpose(img.FLIP_TOP_BOTTOM)), (320,240), interpolation=cv.INTER_AREA))
    t_image2 = Image.fromarray(img_bytes[2])
    cv.imshow("Camera3", cv.resize(np.array(t_image2.convert('L').transpose(img.FLIP_TOP_BOTTOM)), (320,240), interpolation=cv.INTER_AREA))
    cv.waitKey(1)
    
    Points = []
    g_freeMemary = c_dll.FreeAllocatedMemary
    g_freeMemary()
    '''
    for i in range(buff3.num_camera):
        for j in range(buff3.width * buff3.height):
                data.append(buff3.data[i][j])
    
    print("Data: ",data.__len__())
    

    Points = []
    data = []
    '''