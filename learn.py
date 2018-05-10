import numpy as np
import cv2
cap = cv2.VideoCapture('test.mp4')

# # ## MOG2
# # # fgbg = cv2.createBackgroundSubtractorMOG2()
# # # while(1):
# # #     ret, frame = cap.read()
# # #     fgmask = fgbg.apply(frame)
# # #     cv2.imshow('frame',fgmask)
# # #     k = cv2.waitKey(30) & 0xff  # ESC de tat
# # #     print(k)
# # #     if k == 27:
# # #         break
# # # cap.release()
# # # cv2.destroyAllWindows()

# # ## GMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while 1:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# # # cap.release()
# # # cv2.destroyAllWindows()

# f = cv2.VideoCapture(0)
# while 1:
    
#     _, cam = f.read()
#     cam = cv2.flip(cam, 1)
#     cv2.imshow('frame',cam)
#     k = cv2.waitKey(30)    
#     if k == 27:
#         break


# import cv2
# import sys
 
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
# if __name__ == '__main__' :
 
#     # Set up tracker.
#     # Instead of MIL, you can also use
    
#     tracker = cv2.TrackerMIL_create()
#     # Read video
#     video = cv2.VideoCapture(0)
 
#     # Exit if video not opened.

 
#     # Read first frame.
#     ok, frame = video.read()
     
#     # Define an initial bounding box
#     # bbox = (287, 23, 86, 320)
 
#     # Uncomment the line below to select a different bounding box
#     # bbox = cv2.selectROI(frame)
 
#     # Initialize tracker with first frame and bounding box
#     # ok = tracker.init(frame, bbox)
 
#     while True:

#         # Read a new frame
#         ok, frame = video.read()

#         # Start timer
#         # timer = cv2.getTickCount()
 
#         # Update tracker
#         # ok, bbox = tracker.update(frame)
 
#         # Calculate Frames per second (FPS)
#         # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
#         # Draw bounding box
#         # if ok:
#         #     # Tracking success
#         # p1 = (int(bbox[0]), int(bbox[1]))
#         # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#         # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#         # else :
#         #     # Tracking failure
#         #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
#         # Display tracker type on frame
#         # cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
     
#         # Display FPS on frame
#         # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
 
#         # Display result

#         r = cv2.selectROI(frame)
#         imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0]+r[2])]
#         cv2.imshow("Tracking", imCrop)
 
#         # Exit if ESC pressed
#         k = cv2.waitKey(1) & 0xff
#         if k == 27 :
#             break

# import cv2
# import numpy as np
 
# if __name__ == '__main__' :
 
#     # Read image
#     im = cv2.imread("ha.jpg")
     
#     # Select ROI
#     r = cv2.selectROI(im)
     
#     # Crop image
#     imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
#     # Display cropped image
#     cv2.imshow("Image", imCrop)
#     cv2.waitKey(0)