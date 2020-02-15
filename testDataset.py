import DataReceiver
import cv2

while 1:
    Camera_Data = DataReceiver.aquireImageData(True,True) 
    for i in Camera_Data[2].keys():
        to_proc_img = Camera_Data[1][int(i)]
        a = (Camera_Data[2])[i]
        for j in range(len((Camera_Data[2])[i])):
            point1 = (Camera_Data[2][i][j][0],Camera_Data[2][i][j][1])
            point2 = ((Camera_Data[2])[i][j][0]+(Camera_Data[2])[i][j][2],(Camera_Data[2])[i][j][1]+(Camera_Data[2])[i][j][3])
            cv2.rectangle(to_proc_img, point1, point2, (0,0,255), 2)
            #cv2.rectangle(to_proc_img, (0,0), (200,400), (0,0,255), 2)
        cv2.imshow(i,to_proc_img)
        cv2.waitKey(1)
    #for i in GT_DATA.keys:
#print(GT_DATA.keys)