import cv2
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI

OPEN_AI_APIKEY = ''

# Define the paths for the YOLO configuration, weights, and classes files
yolo_config = './YOLO/yolov3.cfg'  # YOLO architecture configuration file
yolo_weights = './YOLO/yolov3.weights'  # YOLO pre-trained weights file
yolo_classes = './YOLO/yolov3.txt'  # File containing the list of detectable classes for YOLO

# Read the classes file and store each class in a list
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Load the YOLO model into memory using OpenCV
redeYOLO = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)

# Function to detect objects in the image using YOLO
def definirObjeto(imagem):
    # Get the output layer names of the neural network
    nomeCamadasRN = redeYOLO.getLayerNames()
    nomeCamadasRN = [nomeCamadasRN[i - 1] for i in redeYOLO.getUnconnectedOutLayers()]
    
    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(
        imagem,
        1 / 255.0,  # Normalize the image
        (416, 416),  # Resize the image to YOLO's expected size
        swapRB=True,  # Convert from BGR to RGB
        crop=False
    )
    redeYOLO.setInput(blob)  # Send the processed image to the network
    saidaCamadasRN = redeYOLO.forward(nomeCamadasRN)  # Perform inference

    # Initialize lists to store detections
    confidences = []
    classIDs = []
    labels = []

    # Iterate over the network outputs to process detections
    for output in saidaCamadasRN:
        for detection in output:
            scores = detection[5:]  # Class probabilities
            classID = np.argmax(scores)  # Identify the class with the highest score
            confidence = scores[classID]  # Confidence score for the detection
            
            # Filter detections with confidence greater than the threshold
            if confidence > 0.6:
                label = classes[classID]  # Get the name of the class
                labels.append(label)
                classIDs.append(classID)
                confidences.append(confidence)
    
    # Return a list of unique detected labels
    return list(set(labels))

# Function to download an image from a URL
def armazenar_imagem(url): 
    try:
        # Download the image from the URL
        resposta = requests.get(url, timeout=10)
        if resposta.status_code == 200 and resposta.content:
            # Convert the image to a format usable by OpenCV
            imagemNP = np.array(bytearray(resposta.content), dtype=np.uint8)
            imagem = cv2.imdecode(imagemNP, cv2.IMREAD_COLOR)
            return imagem
        else:
            print(f"Error downloading the image: {url} - HTTP Code: {resposta.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Exception while downloading the image: {url} - Error: {e}")
        return None

# Function to check relevance between a title and detected labels
def ehRelacionado(titulo, labels):
    # Create the client to call the OpenAI API
    client = OpenAI(api_key=OPEN_AI_APIKEY)
    
    # Make a request to the GPT model to determine relevance
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent assistant that checks the relevance of an image to an advertisement. Receive the ad title and the categorization of the image done by the YOLO model. Your task is to evaluate if at least one of the labels matches the ad title and respond with 'true' or 'false'."
            },
            {
                "role": "user",
                "content": f """
                Advertisement title: "{titulo}"
                Image categorization by YOLO (labels): "{','.join(labels)}"
                """
            }
        ]
    )
    return completion.choices[0].message  # Return the response from the model

# Function to extract information from a website using Selenium
def pegarInfo(link, driver):
    # Access the webpage from the provided link
    driver.get(link)
    
    # Wait until all <img> elements are loaded
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'img')))
    anuncio = driver.find_elements(By.TAG_NAME, 'img')
    
    # Collect the 'alt' and 'src' attributes of each image
    titulo = [img.get_attribute('alt') for img in anuncio]
    imagemUrl = [img.get_attribute('src') for img in anuncio]
    
    return (titulo, imagemUrl)  # Return titles and image URLs

# Main block
if __name__ == '__main__':
    # Initialize the Edge browser
    driver = webdriver.Edge()
    
    try:
        # Receive the advertisement URL from the user
        url = input('Enter the advertisement link: ')
        
        # Extract titles and image URLs from the webpage
        titulos, imagemUrls = pegarInfo(url, driver)
        
        # Process each title and image URL
        for i, (titulo, imagemUrl) in enumerate(zip(titulos, imagemUrls), start=1):
            print(f"\nProcessing item {i}...")
            
            # Download the image from the URL
            imagem = baixar_imagem(imagemUrl)
            if imagem is None:
                print(f"Image {i}: Could not download the image.")
                continue
            
            # Detect objects in the image
            outputLabel = definirObjeto(imagem)
            if len(outputLabel) == 0:
                print(f"Image {i}: Could not identify any objects.")
                continue
            
            # Check relevance between the title and detected objects
            resultado = ehRelacionado(titulo, outputLabel)
            
            # Display the result
            print(f"Result for item {i}:")
            print(f"Title: {titulo}")
            print(f"Output Label: {outputLabel}")
            print(f"Relevance: {resultado.content}")

    finally:
        # Close the browser at the end of the execution
        driver.quit()
