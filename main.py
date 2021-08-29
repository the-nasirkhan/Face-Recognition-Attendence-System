import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = "images"
images = []
person_Names = []
myList = os.listdir(path)
print(myList)

for curr_image in myList:
    current_images = cv2.imread(f"{path}/{curr_image}")
    images.append(current_images)
    person_Names.append(os.path.splitext(curr_image)[0])
print(person_Names)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name):
    with open("Attendance.csv",'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            timeStr = time_now.strftime("%H:%M:%S")
            dateStr = time_now.strftime("%d/%m/%Y")
            f.writelines(f"\n{name},{timeStr},{dateStr}")

encodeListKnown = faceEncodings(images)
print('All Encoding Complete!!!!')

cature = cv2.VideoCapture(0)

while True:
    ret, frame = cature.read()
    faces = cv2.resize(frame,(0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFaces, faceLoc in zip (encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFaces)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFaces)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = person_Names[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 2)
            cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            attendance(name)

    cv2.imshow("webcame", frame)
    if cv2.waitKey(1) == 13:
        break

cature.release()
cv2.destroyAllWindows()