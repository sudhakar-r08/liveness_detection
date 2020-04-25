import cv2


cap = cv2.VideoCapture()
cap.open("rtsp://admin:admin@12345@192.168.1.102:554/Streaming/channels/1/")

while True:

    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()