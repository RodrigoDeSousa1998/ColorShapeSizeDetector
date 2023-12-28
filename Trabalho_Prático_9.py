# TP9 - Seguimento de uma forma e/ou cor em particular, com eventual mudança de cor em tempo real.
# 
# Proposto: Identificar e contar esferas laranja do FUN CHALLENGE
#    Extra: Contar apenas as esferas que se encontram dentro das linhas verdes do campo
#
# Trabalho realizado por: Rodrigo de Sousa 
#                         Manuel Rodrigues
#                         

###################################### TO DO LIST ############################################

#! TODO Meter contador de bolas de um  lado e de outro na interface
#! TODO Fazer deteções de linhas verdes e só contar circulos dentro das linhas verdes (ver se 
#       há linha de meio campo, para contar ou só as bolas de uma lado, ou do outro)
#! TODO Fazer videos com bolas a mexer-se dentro de campo no onenote (ou outra aplicação)
#! TODO Eliminar ruidos depois da calibração de cor, através de métricas de áreas da imagem
#! TODO Fazer menu com teclas de comando para UX (para recablibrar, mudar raio de bolas etc)
#! TODO Fazer funções para compactar código

####################################### DEVELOPMENT ##########################################

import numpy as np
import cv2

# Calibrations Initial Values
MaxH = 1
MinH = 1
MaxR = 1
MinR = 1

# Getting video feed from camera
#videoFeed = cv2.VideoCapture(1)

imageSelected = cv2.imread("ImagemTeste1.png")

# To create trackbar windows
halfTable = cv2.imread("UI_Image1.png")
fullTable = cv2.imread("UI_Image2.png")

def fieldMap(x): 
    temp=halfTable.copy()
    cv2.rectangle(temp, (int(MinH*6.42), 0), (int(MaxH*6.42), 140), (127, 127, 127), 2)
    cv2.imshow(window_name_1, temp)

def ballMap(x): 
    temp=fullTable.copy()
    cv2.rectangle(temp, (int(MinH*6.42), 0), (int(MaxH*6.42), 140), (127, 127, 127), 2)
    cv2.circle(temp, (569, 380), int(MinR), (255,0,0), 2)
    cv2.circle(temp, (569, 760), int(MaxR), (255,0,0), 2)
    cv2.imshow(window_name_2, temp)

window_name_1 = 'Field Line Calibrations'
cv2.namedWindow(window_name_1)
cv2.createTrackbar('MinH',window_name_1,10,179, fieldMap)
cv2.createTrackbar('MaxH',window_name_1,10,179, fieldMap)

window_name_2 = 'Ball Calibrations'
cv2.namedWindow(window_name_2)
cv2.createTrackbar('MinH',window_name_2,10,179, ballMap)
cv2.createTrackbar('MaxH',window_name_2,10,179, ballMap)
cv2.createTrackbar('MinR',window_name_2,10,200, ballMap)
cv2.createTrackbar('MaxR',window_name_2,10,200, ballMap)

#Displaying the feed
# ret, frame = videoFeed.read()
# while (ret and cv2.waitKey(1)!=27):
while (cv2.waitKey(1)!=27):
    
    # Conversion to HSV
    hsv = cv2.cvtColor(imageSelected, cv2.COLOR_BGR2HSV)

    # Field Area Detection
    # Filter lower and upper bound
    MinH = cv2.getTrackbarPos('MinH', window_name_1)
    MaxH = cv2.getTrackbarPos('MaxH', window_name_1)
    min_color = np.array([MinH, 20, 20])
    max_color = np.array([MaxH, 255, 255])
    FAD_filtered = cv2.inRange(hsv, min_color, max_color)

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Opening to remove noise from the feed
    opening = cv2.morphologyEx(FAD_filtered, cv2.MORPH_OPEN, kernel)
    # Closing to have clear edges
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    FAD_denoised = closing

    cv2.imshow('result', FAD_denoised)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(FAD_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    # Iterate through the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Check if the polygon has four vertices (a square)
        if len(approx) == 4:
            # Draw a bounding box around the square
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region inside the square
            region_of_interest = imageSelected[y:y+h, x:x+w]
            cv2.imshow('result2', region_of_interest)

            MinH = cv2.getTrackbarPos('MinH', window_name_2)
            MaxH = cv2.getTrackbarPos('MaxH', window_name_2)
            min_color = np.array([MinH, 20, 20])
            max_color = np.array([MaxH, 255, 255])
            BD_filtered = cv2.inRange(region_of_interest, min_color, max_color)

            MinR = cv2.getTrackbarPos('MinR', window_name_2)
            MaxR = cv2.getTrackbarPos('MaxR', window_name_2)
            MinDistCircles = 100
            circles = cv2.HoughCircles(BD_filtered,cv2.HOUGH_GRADIENT,1,MinDistCircles, 
                                            param1=MinR,param2=MaxR,minRadius=0,maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv2.circle(BD_filtered,(i[0],i[1]),i[2],(255,0,0),2)
                    cv2.circle(BD_filtered,(i[0],i[1]),2,(0,255,255),2)    
                number_circles = circles.shape[1]
                print(number_circles)

            cv2.imshow('detected circles',BD_filtered)

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