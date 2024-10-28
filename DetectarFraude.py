import cv2
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI

OPEN_AI_APIKEY = ''

yolo_config = './YOLO/yolov3.cfg'
yolo_weights = './YOLO/yolov3.weights'
yolo_classes = './YOLO/yolov3.txt'

with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

redeYOLO = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)


def definirObjeto(imagem):
    nomeCamadasRN = redeYOLO.getLayerNames()
    nomeCamadasRN = [
        nomeCamadasRN[i - 1] for i in redeYOLO.getUnconnectedOutLayers()
    ]
    blob = cv2.dnn.blobFromImage(
        imagem,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )
    redeYOLO.setInput(blob)
    saidaCamadasRN = redeYOLO.forward(nomeCamadasRN)
    confidences = []
    classIDs = []
    labels = []

    for output in saidaCamadasRN:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.6:
                label = classes[classID]
                labels.append(label)
                classIDs.append(classID)
                confidences.append(confidences)
    return list(set(labels))


def baixar_imagem(url):
    resposta = requests.get(url)
    imagemNP = np.array(bytearray(resposta.content), dtype=np.uint8)
    imagem = cv2.imdecode(imagemNP, cv2.IMREAD_COLOR)
    return imagem


def ehRelacionado(titulo, labels):

    client = OpenAI(api_key=OPEN_AI_APIKEY)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente inteligente que verifica a relevância de uma imagem em relação a um anúncio. Receba informações sobre o título e a categorização da imagem feita pelo modelo YOLO. Sua tarefa é avaliar se pelo menos uma das labels combina com o título  do anúncio e responder apenas com 'true' ou 'false'."
            },
            {
                "role": "user",
                "content": f """
                Título do anúncio: "{titulo}"
                Categorização da imagem pelo YOLO (labels): "{','.join(labels)}"
    """
            }
        ]
    )
    return completion.choices[0].message


def pegarInfo(link, driver):
    driver.get(link)

    h1 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'h1'))
    )

    imagens = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'image'))
    )

    titulo = h1.text
    imagemUrl = imagens.get_attribute('src')

    return (titulo, imagemUrl)


if __name__ == '__main__':
    driver = webdriver.Edge()

    url = input('Qual o link do anúnico:')
    titulo, imagemUrl = pegarInfo(url, driver)
    imagem = baixar_imagem(imagemUrl)

    if imagem is None:
        raise Exception("Não foi possível baixar a imagem.")

    outputLabel = definirObjeto(imagem)
    if len(outputLabel) == 0:
        raise Exception("Não foi possível identificar o objeto.")

    resultado = ehRelacionado(titulo, outputLabel)
    print(resultado.content)
    print(outputLabel)
