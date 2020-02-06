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

class car_info(Structure):
    _fields_= [("speed", c_float),("car_position_x", c_float)
    ,("car_position_y", c_float), ("car_position_z", c_float)]

class AlgInformation(Structure):
    _fields_= [("steeringMechanism_TuringRate", c_float),("brakingMechanism_BrakingRate", c_float)
    ,("engineMechanism_ThrottleRate", c_float), ("message", c_char*200)]

Points = []

c_dll = cdll.LoadLibrary("SM_Access.dll")
g_carPos = c_dll.GetCarPosition
g_carPos.restype = Position
g_PCData = c_dll.GetPointCloudData
g_PCData.restype = PointCloudPackage
g_CameraData =c_dll.GetCarCameraData
g_CameraData.restype = VideoStreamPack
g_FreePointCloudData = c_dll.FreePointCloudMemary
g_freeMemary = c_dll.FreeAllocatedMemary
g_CarInformation = c_dll.GetCarInformation
g_CarInformation.restype = car_info
w_CarControllerCommand = c_dll.writeAlgControllerCommand


hasShown = 0

def _packeageTestFunction():
    
    buff1 = g_carPos()
    #print("Car position: ",buff1.x," ",buff1.y," ",buff1.z)

    PC_Origin = []
    for i in range(12):
        buff2 = g_PCData()
        

        PC_Origin = np.ctypeslib.as_array(cast(buff2.pointCloudInfo, POINTER(c_float))
        , (buff2.dataLength, 4))

        g_FreePointCloudData()
        
        
    #print("Points: ",buff2.dataLength)

    buff3 = g_CameraData()

    data = []
    img_bytes = []
    for i in range(buff3.num_camera):
        img_origin = np.ctypeslib.as_array(cast(buff3.data[i], POINTER(c_ubyte))
        , (buff3.height, buff3.width, 4))
        #img_buf = np.delete(img_origin, -1, axis=1)
        img_bytes.append(img_origin)

    global hasShown
    if(hasShown == 2):#change to 0 when debug
        t_image = Image.fromarray(img_bytes[0]).convert('RGB')
        Image._show(Image.fromarray(np.array(cv.cvtColor(np.array(t_image),cv.COLOR_BGR2RGB))))
        t_image1 = Image.fromarray(img_bytes[1]).convert('RGB')
        Image._show(Image.fromarray(np.array(cv.cvtColor(np.array(t_image1),cv.COLOR_BGR2RGB))))
        t_image2 = Image.fromarray(img_bytes[2]).convert('RGB')
        Image._show(Image.fromarray(np.array(cv.cvtColor(np.array(t_image2),cv.COLOR_BGR2RGB))))
        hasShown = 1
    
    t_image = Image.fromarray(img_bytes[0])
    cv.imshow("Camera1", cv.resize(np.array(t_image), (320,240), interpolation=cv.INTER_AREA))
    t_image1 = Image.fromarray(img_bytes[1])
    cv.imshow("Camera2", cv.resize(np.array(t_image1), (320,240), interpolation=cv.INTER_AREA))
    t_image2 = Image.fromarray(img_bytes[2])
    cv.imshow("Camera3", cv.resize(np.array(t_image2), (320,240), interpolation=cv.INTER_AREA))
    cv.waitKey(1)
    
    Points = []
    
    g_freeMemary()
    '''
    for i in range(buff3.num_camera):
        for j in range(buff3.width * buff3.height):
                data.append(buff3.data[i][j])
    
    print("Data: ",data.__len__())
    

    Points = []
    data = []
    '''

"""Return the Car camera gray image data and the car position"""
def aquireImageData(isGray = False):
    out_CarPosition = []
    out_CameraImage = []

    buff1 = g_carPos()
    out_CarPosition = [buff1.x,buff1.y,buff1.z]

    buff3 = g_CameraData()
    img_bytes = []
    for i in range(buff3.num_camera):
        img_origin = np.ctypeslib.as_array(cast(buff3.data[i], POINTER(c_ubyte))
        , (buff3.height, buff3.width, 4))
        #img_buf = np.delete(img_origin, -1, axis=1)
        img_bytes.append(img_origin)
    t_image1 = Image.fromarray(img_bytes[0]).convert('RGB')
    t_image2 = Image.fromarray(img_bytes[1]).convert('RGB')
    t_image3 = Image.fromarray(img_bytes[2]).convert('RGB')

    if isGray:
        out_CameraImage = [np.array(cv.cvtColor(np.array(t_image1),cv.COLOR_BGR2GRAY)),
        np.array(cv.cvtColor(np.array(t_image2),cv.COLOR_BGR2GRAY)),
        np.array(cv.cvtColor(np.array(t_image3),cv.COLOR_BGR2GRAY))]
    else:
        out_CameraImage = [np.array(cv.cvtColor(np.array(t_image1),cv.COLOR_BGR2RGB)),
        np.array(cv.cvtColor(np.array(t_image2),cv.COLOR_BGR2RGB)),
        np.array(cv.cvtColor(np.array(t_image3),cv.COLOR_BGR2RGB))]

    g_freeMemary = c_dll.FreeAllocatedMemary
    g_freeMemary()

    return out_CarPosition,out_CameraImage


"""Return the Lidar point cloud data and the car position"""
def aquirePointCloudData():
    out_CarPosition = []
    out_PointCloud = []

    buff1 = g_carPos()
    out_CarPosition = [buff1.x,buff1.y,buff1.z]

    for i in range(12):
        buff2 = g_PCData()
        out_CarPosition = np.ctypeslib.as_array(cast(buff2.pointCloudInfo, POINTER(c_float))
        , (buff2.dataLength, 4))
        out_CarPosition = np.delete(out_CarPosition, -1, axis=1)

        g_FreePointCloudData()

    return out_CarPosition,out_PointCloud

"""Return the car position"""
def aquireCarPosition():
    out_CarPosition = []

    buff1 = g_carPos()
    out_CarPosition = [buff1.x,buff1.y,buff1.z]

    return out_CarPosition

def aquireCarInformation():
    buff1 = g_CarInformation()

    out_speed = buff1.speed
    return out_speed

def writeAlgControllerSharedMemary(in_command):
    writeInfo = AlgInformation()
    writeInfo.steeringMechanism_TuringRate = in_command[0]
    writeInfo.brakingMechanism_BrakingRate = in_command[1]
    writeInfo.engineMechanism_ThrottleRate = in_command[2]
    w_CarControllerCommand(writeInfo)

if __name__ == "__main__":
    while True:
        #_packeageTestFunction()
        print(aquireCarInformation())
    """_,img = aquireImageData()
    Image.fromarray(img[0]).save("Output1.jpg")
    Image.fromarray(img[1]).save("Output2.jpg")
    Image.fromarray(img[2]).save("Output3.jpg")"""
    #print(aquireCarPosition())
    #print(aquirePointCloudData())