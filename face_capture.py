import cv2
import sys
import pickle
import face_recognition
import argparse
import datetime
import math

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

nameStatus = {}
nameLastUpdate = {}
enteringTime = {}

unknowncount = 0
now = datetime.datetime.now()

logfilename = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '_logs.txt'
print(logfilename)
logfile = open(logfilename,'w')

def snap_shot(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
	    model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        print(name,' detected')
        if name is "Unknown":
            name = name + str(unknowncount)
            unknowncount += 1
        #Person Detected 
        #Do all operations here

        currentTime = datetime.datetime.now()
        
        if not name in nameStatus:
            logfile.write(name+' has entered at - ' + str(currentTime.hour) + ':' + str(currentTime.minute) + ':' + str(currentTime.second) + '\n')
            print(name+' has entered')
            nameStatus[name]='IN'
            nameLastUpdate[name]=currentTime
            enteringTime[name]=currentTime
        
        else:
            lastUpdateTime = nameLastUpdate[name]
            difference = currentTime - lastUpdateTime
            if difference.seconds < 10:
                print('last_update_time_too_close : not changing status')   
            else:
                if nameStatus[name] == 'IN':
                    timein = currentTime - enteringTime[name]
                    minutesIn = math.floor(timein.seconds /60)
                    secondsIn = timein.seconds %60
                    logfile.write(name+' has left at - ' + str(currentTime.hour) + ':' + str(currentTime.minute) + ':' + str(currentTime.second))
                    logfile.write('...........Duration - ' + str(minutesIn) + ' mins ' + str(secondsIn) + 'secs \n' )
                    print(name+' has left.')
                    nameStatus[name] = 'OUT'
                else:
                    logfile.write(name+' has entered at - ' + str(currentTime.hour) + ':' + str(currentTime.minute) + ':' + str(currentTime.second) + '\n')
                    print(name+' has entered.')
                    nameStatus[name]='IN'
                    enteringTime[name]=currentTime
            
            nameLastUpdate[name] = currentTime


        #Drawing rectangle   
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		    0.75, (0, 255, 0), 2)

    #cv2.imwrite("Image_boxed_"+str(count)+".jpg", image)
    #print("saved image succesfully as Image_boxed_"+str(count)+".jpg")

###
###
###     MAIN PROGRAM
###     BEGINS HERE
###       
###


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open('encodings.pickle', "rb").read())

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

print("\n\n~~~~~~~~~~~~~~~~~~~CAPTURING VIDEO TO DETECT FACES~~~~~~~~~~~~~~~~~~~~~~~\n\n")
video_capture = cv2.VideoCapture(0)
count =0 
framenumber=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    print(framenumber)
    framenumber+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if not len(faces) is 0:
        #print('face found'+str(count))
        count+=1
        snap_shot(frame)
        #if count is 4:
        #    break

    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
        #print(x,y,h,w)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()