import cv2
import dlib

imagem = cv2.imread("data\\fotos\\grupo.0.jpg")
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

facesDetectadas = detector(imagem, 1)
print(facesDetectadas)

for face in facesDetectadas:
    e, t, d, b = (int(face.rect.left())), (int(face.rect.top())), (int(face.rect.right())), (int(face.rect.bottom()))
    c = face.confidence
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

cv2.imshow("CNN detector", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
