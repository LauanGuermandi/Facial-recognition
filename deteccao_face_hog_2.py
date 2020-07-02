import cv2
import dlib

subdetector = ["Olhar a frente", "Vista a esquerda", "Vista a direita", "A frente virando para a esquerda",
               "A frente virando para a direita"]

imagem = cv2.imread("data\\fotos\\grupo.0.jpg")
detector = dlib.get_frontal_face_detector()

# 0.3 é o minimo de potuacao que a imagem pode ter
facesDetectadas, pontuacao, idx = detector.run(imagem, 1, 0.3)
print(facesDetectadas, pontuacao, idx)

for i, face in enumerate(facesDetectadas):
    print("Detecao: {}, pontuacao: {}, Subdetector: {}".format(face, pontuacao[i], subdetector[idx[i]]))
    e, t, d, b = (int(face.left())), (int(face.top())), (int(face.right())), (int(face.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

cv2.imshow("HOG detector", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
