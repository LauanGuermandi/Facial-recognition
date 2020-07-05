import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFaces = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("data\\recursos\\shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join("data\\fotos\\treinamento", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFaces(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)
    print("Numros de faces: " + str(numeroFacesDetectadas))

    if numeroFacesDetectadas > 1:
        print("Há mais de uma face encontrada!")
        exit(0)
    elif numeroFacesDetectadas < 1:
        print("Nenhuma face encontrada!")
        exit(0)

    # Face é um bounding box
    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorDacial = [df for df in descritorFacial]

        npArrayDescritorFacial = np.asarray(listaDescritorDacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        indice[idx] = arquivo
        idx += 1

np.save("descritores_rn.npy", descritoresFaciais)
with open("indices_rn.pickle", "wb") as f:
    cPickle.dump(indice, f)

