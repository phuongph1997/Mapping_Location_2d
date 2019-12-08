import numpy as np
import cv2
import time
import glob
import re
import matplotlib.pyplot as plt
def GetAbsoluteScale(f, frame_id):
      x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
      x    , y    , z     = f[frame_id][3], f[frame_id][7], f[frame_id][11]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale
      
def FeatureTracking(img_1, img_2, p1):

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2


def FeatureDetection():
#     thresh = dict(threshold=25, nonmaxSuppression=True);
#      fast = cv2.FastFeatureDetector_create(**thresh)
#     return fast
#    orb = cv2.ORB_create(edgeThreshold=25, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
#    orb=cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create(firstLevel=0, nfeatures=500)
#     return sift
    return orb

def NumericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts 
def GetFeature():
    
    for filename in sorted(glob.glob('../Data_Train/*.npy'), key=NumericalSort):
        coor = []
        contents = np.load(filename)
        coor = filename[-11:-4].split("-")
        List_des.append(contents)
        List_coor.append(coor)
    print(len(List_coor))
    

def Measure_des(des):
    global List_matches
    num = 0
    list_temp = []
    #print("is range {0}".format(Is_range))
    if Is_range == True :
        List_matches = []
        m0 = time.time()
        compare_feature(des, List_des[Start_range:End_range], Start_range,End_range)
        print(time.time()-m0)
        print("o tren")
        if len(List_matches) !=0:
            return
    while num <= 150:
        mat = bf.match(des, List_des[num])
        list_temp.append(len(mat))
        num+=50
    #print(list_temp)
    #print ("max = {0}".format(max(list_temp)))
    # compare_feature(des, List_des[0:213], 0,213)
    if max(list_temp) == list_temp[0]:
        print('khoang 0')
        compare_feature(des, List_des[0:51], 0,51)
    elif max(list_temp) == list_temp[1]:
        print('khoang 50')
        compare_feature(des, List_des[50:101], 50,101)
    elif max(list_temp) == list_temp[2]:
        print('khoang 100')
        compare_feature(des, List_des[100:151], 100,151)
    elif max(list_temp) == list_temp[3]:
        print('khoang 150')
        compare_feature(des, List_des[150:], 150,len(List_des))

    
        
def compare_feature(des, subListDes, start,end):
    global Start_range, End_range, Is_range, mang_tam
    intSubStart = start
    print('start {0}'.format(start))
    print('end {0}'.format(end))
    t1 = time.time()
    for des2 in subListDes:
        matches = bf.match(des, des2)
        mang_tam.append(len(matches))
        if len(matches) > 450:
            List_matches.append(List_coor[start])
            Start_range = start
            End_range = end + (start-intSubStart)
            Is_range = True
            break
        start+=1
    print("time {0}".format(time.time()-t1))
        
        
def Train(des, x, y, intName):
    file_name = "../Data_Train/IMG" + str(intName)  + "_" + str(x) + "-" + str(y) 
    print(file_name)
    if des is not None:
        np.save(file_name, des)
#initialization
#####Begin#####

cap = cv2.VideoCapture('../VideoTest/video.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
f, img_1 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
f, img_2 = cap.read()
if len(img_1) == 3:
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
else:
    gray_1 = img_1
    gray_2 = img_2
#find the detector
detector = FeatureDetection()
kp1, des1      = detector.detectAndCompute(img_1, None)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = FeatureTracking(gray_1, gray_2, p1)

#Camera parameters
fc = 554.122
pp = ( 246.55, 307.60)
E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);

#initialize some parameters
preFeature = p2
preImage   = gray_2
R_f = R
t_f = t
Map2d = np.zeros((1200, 1200, 3), dtype=np.uint8)
CountFrameIgnor = 0
#play image sequences
Is_range = False
Start_range = 0
End_range = 0
MIN_NUM_FEAT  = 200
img=0
List_matches = []
mang_tam = []
List_coor = []
List_des = []
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   
CountFrameTraining = 0 
#####END#####
## call funtion initially
GetFeature()
##### <<<BEGIN>>> #####

while(True):
    start = time.time()
    localtion = cv2.imread('mapold.png')
    ret, FrameTemp = cap.read()
    if ret !=True:
        print("can not get the frame")
        break
    if CountFrameIgnor ==0:
        img= FrameTemp
    CountFrameIgnor+=1
    if CountFrameIgnor == 5:
        CountFrameIgnor =0
       
        if (len(preFeature) < MIN_NUM_FEAT):
            feature, des2   = detector.detectAndCompute(preImage, None)
            preFeature = np.array([ele.pt for ele in feature],dtype='float32')
        curImage_c = img    
        if len(curImage_c) == 3:
              curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)
        else:
              curImage = curImage_c
        
        kp1, des3 = detector.detectAndCompute(curImage, None);
        Measure_des(des3)
        # print(mang_tam)
        # print("max {0}".format(max(mang_tam)))
        #plt.bar(range(0,214), [0,1,2,3])
        #plt.bar(np.arange(len(mang_tam)), mang_tam, align='center', alpha=0.5)
        #plt.pause(0.02)
        mang_tam = []
#         compare_feature(des3)
        # print(max(List_matches,default=0))
        #List_matches= []
        ########### ve map   
        # preFeature, curFeature = FeatureTracking(preImage, curImage, preFeature)
        # E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0); 
        # _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp);
        # t_f = t_f + 1*R_f.dot(t)    
        # R_f = R.dot(R_f)   
        # preImage = curImage
        # preFeature = curFeature
        

        ####Visualization of the result
        # draw_x, draw_y = int(t_f[0]) + 300, int(t_f[2]) + 300;
        ############## end

        if len(List_matches)!=0:
            print("day ne {0}".format(List_matches))
            draw_x, draw_y =  int(List_matches[0][0]),int(List_matches[0][1])
            # List_matches= []
#         print(draw_x)
#         print(draw_y)
        #save description of frame to database
        # Train(des3, draw_x, draw_y, CountFrameTraining)
        # CountFrameTraining +=1
            cv2.circle(localtion, (draw_x, draw_y) ,1, (0,255,255), 2);    
        # text = "khoang cach so voi land mark: x ={0:02f}m y = {1:02f}m".format(float(500-draw_x), float(500-draw_y));
        # temp = np.zeros((480, 1280, 3), dtype=np.uint8)
    
        cv2.imshow( "Map2dectory", localtion )
        cv2.imshow("anh test ",img) 
        stop = time.time()
        print(stop - start)
   
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
plt.show()
#cv2.imwrite('map.png', Map2d);
cv2.destroyAllWindows()
print("DONE!!!!!")