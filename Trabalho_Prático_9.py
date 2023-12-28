# TP9 - Seguimento de uma forma e/ou cor em particular, com eventual mudança de cor em tempo real.
# 
# Proposto: Identificar e contar esferas laranja do FUN CHALLENGE
#    Extra: Contar apenas as esferas que se encontram dentro das linhas verdes do campo
#
# Trabalho realizado por: Rodrigo de Sousa 
#                         Manuel Rodrigues
#                         

###################################### TO DO LIST ############################################

#! TODO Meter contador de bolas de um  lado da imagem analisada com circulos à volta
#! TODO Fazer deteções de linhas verdes e só contar circulos dentro das linhas verdes (ver se 
#       há linha de meio campo, para contar ou só as bolas de uma lado, ou do outro)
#! TODO Fazer videos com bolas a mexer-se dentro de campo
#! TODO Eliminar ruidos depois da calibração de cor, através de métricas de áreas da imagem
#! TODO Meter hierarchy para identificar apenas os contornos dos maior quadrado verde que encontrar
#! TODO Fazer menu com teclas de comando para alternar a contagem de bolas de um lado e outro do campo 
#! TODO Fazer menu com teclas de comando para UX (para recablibrar, mudar raio de bolas etc)
#! TODO Fazer funções para compactar código

####################################### DEVELOPMENT ##########################################

import numpy as np
import cv2

# Calibrations Initial Values
FieldUT    = 1
FieldLT    = 1
BallUT     = 1
BallLT     = 1
MaxRadius  = 1
MinRadius  = 1
AccumThsld = 10

# Variables Initial Values
contours = None

# Getting video feed from camera
#videoFeed = cv2.VideoCapture(1)

image = cv2.imread("ImagemTeste1.png")

# To create trackbar windows
table = cv2.imread("UI_Image.png")

def Map(x): 
    temp=table.copy()
    cv2.rectangle(temp, (int(BallLT *6.42), 0), (int(BallUT *6.42), 140), (127, 127, 127), 2)
    cv2.rectangle(temp, (int(FieldLT*6.42), 0), (int(FieldUT*6.42), 140), (127, 127, 127), 2)
    cv2.circle(temp, (380, 254), int(MinRadius*0.5), (255,0,0), 2)
    cv2.circle(temp, (760, 254), int(MaxRadius*0.5), (255,0,0), 2)
    cv2.imshow(window_name, temp)

window_name = 'Calibrations'
cv2.namedWindow(window_name)
cv2.createTrackbar('Field LT',window_name,1,179, Map)
cv2.createTrackbar('Field UT',window_name,10,179, Map)
cv2.createTrackbar('Ball LT',window_name,1,179, Map)
cv2.createTrackbar('Ball UT',window_name,10,179, Map)
cv2.createTrackbar('Min Radius',window_name,1,200, Map)
cv2.createTrackbar('Max Radius',window_name,10,200, Map)
cv2.createTrackbar('Acc.',window_name,10,200, Map)

#Displaying the feed
# ret, frame = videoFeed.read()
# while (ret and cv2.waitKey(1)!=27):
while (cv2.waitKey(1)!=27):
    
    # Working Image
    if contours is None:
        temp=image.copy()
        cv2.imshow('Detected Circles',temp)

    # Conversion to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Field Area Detection
    # Filter lower and upper bound
    FieldLT = cv2.getTrackbarPos('Field LT', window_name)
    FieldUT = cv2.getTrackbarPos('Field UT', window_name)
    FieldMinColor = np.array([FieldLT, 20, 20])
    FieldMaxColor = np.array([FieldUT, 255, 255])
    FAD_filtered = cv2.inRange(hsv, FieldMinColor, FieldMaxColor)

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Opening to remove noise from the feed
    opening = cv2.morphologyEx(FAD_filtered, cv2.MORPH_OPEN, kernel)
    # Closing to have clear edges
    FAD_denoised = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(FAD_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Check if the polygon has four vertices (a square)
        if len(approx) == 4:
            # Draw a bounding box around the square
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region inside the squares
            region_of_interest = hsv[y:y+h, x:x+w]
            blurred_ROI = cv2.medianBlur(region_of_interest, 9)

            # Ball Detection
            # Filter lower and upper bound
            BallLT = cv2.getTrackbarPos('Ball LT', window_name)
            BallUT = cv2.getTrackbarPos('Ball UT', window_name)
            BallMinColor = np.array([BallLT, 20, 20])
            BallMaxColor = np.array([BallUT, 255, 255])
            BD_filtered = cv2.inRange(blurred_ROI, BallMinColor, BallMaxColor)

            MinRadius = cv2.getTrackbarPos('Min Radius', window_name)
            MaxRadius = cv2.getTrackbarPos('Max Radius', window_name)
            AccumThsld = cv2.getTrackbarPos('Acc.', window_name)
            MinDistCircles = 100
            circles = cv2.HoughCircles(BD_filtered,cv2.HOUGH_GRADIENT,1,MinDistCircles, 
                                            param1=10,param2=AccumThsld,minRadius=MinRadius,maxRadius=MaxRadius)

            temp=image.copy()[y:y+h, x:x+w]
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv2.circle(temp,(i[0],i[1]),i[2],(255,0,0),2)
                    cv2.circle(temp,(i[0],i[1]),2,(0,255,255),2)    
                number_circles = circles.shape[1]
                print(number_circles)
            cv2.imshow('Detected Circles',temp)
        else:
            contours = None 

#     ret, frame = videoFeed.read()
# videoFeed.release()

cv2.waitKey(0)
cv2.destroyAllWindows()

################################ CODE TO IMPLEMENT IF NEEDED #################################

# def write_text_to_image(image):
# #To write text in the feed
#     #height = np.size(image, 0)
#     #width = np.size(image, 1)
#     #whiteBackground = np.full((0.3*height,0.3*width), 255)
#     #coordinates = (0, height-100)
#     temp=image.copy()
#     font = cv2.FONT_HERSHEY_COMPLEX
#     fontScale = 1
#     color = (0, 0, 0) #Black
#     lineThickness = 2
#     text = 'PLEASE CALIBRATE FOR ORANGE'
#     cv2.putText(temp, text, (10,10), font, fontScale, color, lineThickness, cv2.LINE_AA)
#     cv2.imshow("image with text",temp)
#     cv2.waitKey(0)

# To vizualize only circles
# colorCircles = cv2.bitwise_and(frame, frame, mask= closing)
# cv2.imshow("Color Circles", colorCircles)