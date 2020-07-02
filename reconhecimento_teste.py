import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFaces = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("data\\recursos\\shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("indices_rn.pickle", allow_pickle=True)
descritoresFaciais = np.load("descritores_rn.npy")
limiar = 0.5

for arquivo in glob.glob(os.path.join("data\\fotos", "*.jpg")):
    imagem = cv2.imread(arquivo)
    # Escala maior para arquivos menores(2)
    facesDetectadas = detectorFaces(imagem, 2)

    for face in facesDetectadas:
        e, t, d, b = (int(face.left())), (int(face.top())), (int(face.right())), (int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        listaDescritorDacial = [df for df in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorDacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = " "

        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)
        texto = "{} {:.4f}".format(nome, distanciaMinima)
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))

    cv2.imshow("HOG detector", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()
