from django.shortcuts import render

# Create your views here.
# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
# import urllib # python 2
import urllib.request # python 3
import json
import cv2
import os
# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			
			image = _grab_image(stream=request.FILES["image"])
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			opc = request.POST.get("opc",None)
			url = request.POST.get("url", None)
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			# load the image and convert
			image = _grab_image(url=url)
			image2 = cv2.imread('image')
		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		faces = detector.detectMultiScale(image, 1.1, 4)
		try:
			for(x, y, w, h) in faces:
				ret = cv2.rectangle(image, (x, y), (x+w, y+h), 2)
			crop = image[y:y+h, x:x+w]
			crop = cv2.medianBlur(crop ,5)
			#cv2.imshow('crop', crop)

			hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 

			# define range of nude color in HSV RED
			red_lower = np.array([161,155,84])
			red_upper = np.array([179,255,255])
		    
		    # Threshold the HSV image to get only red colors
			mask = cv2.inRange(crop, red_lower, red_upper)
			pixelBlackRed = cv2.countNonZero(mask)
		    #print("Total preto na imagem vermelha",pixelBlackRed)

		    # Bitwise-AND mask and original image
			resRed = cv2.bitwise_and(hsv,hsv, mask=mask)
			countRed = 0
			black = np.array([0,0,0])

			# define range of nude color in HSV GREEN
			green_lower = np.array([25,52,72])
			green_upper = np.array([102,255,255])
	    
		    # Threshold the HSV image to get only GREEN colors
			maskGreen = cv2.inRange(crop, green_lower, green_upper)
		    
		    # Bitwise-AND mask and original image
			resGreen = cv2.bitwise_and(hsv,hsv, mask=maskGreen)
			pixelBlackGreen = cv2.countNonZero(maskGreen)


			# define range of nude color in HSV ORANGE
			orange_lower = np.array([0,80,153])
			orange_upper = np.array([153,204,255])
		    
		    # Threshold the HSV image to get only ORANGE colors
			maskOrange = cv2.inRange(crop, orange_lower, orange_upper)
			pixelBlackOrange = cv2.countNonZero(maskOrange)
		   
		    # Bitwise-AND mask and original image
			resOrange = cv2.bitwise_and(hsv,hsv, mask=maskOrange)
			#opc = '5'
			if(pixelBlackRed > pixelBlackGreen and pixelBlackRed > pixelBlackOrange and opc == '1'):
				pele = "Provalvemente tem a pele muito clara"
			elif(pixelBlackOrange > pixelBlackGreen and pixelBlackRed > pixelBlackGreen and pixelBlackOrange > pixelBlackRed and opc == '1'):
				pele = "Provalvemente tem a pele muito clara"
			elif(pixelBlackRed > pixelBlackGreen and pixelBlackRed > pixelBlackOrange and opc == '2'):
				pele = "Provalvemente tem a pele clara"
			elif(pixelBlackOrange > pixelBlackGreen and pixelBlackRed > pixelBlackGreen and pixelBlackOrange > pixelBlackRed and opc == '2'):
				pele = "Provalvemente tem a pele clara"
			elif(pixelBlackOrange > pixelBlackRed and pixelBlackOrange > pixelBlackGreen and opc == '3'):
				pele = "Provalvemente tem a pele clara e cabelos escuros"
			elif(pixelBlackOrange > pixelBlackRed and pixelBlackOrange > pixelBlackGreen and opc == '4'):
				pele = "Provalvemente tem a pele moderadamente pigmentada (Morena)"
			elif(pixelBlackGreen > pixelBlackRed and pixelBlackGreen > pixelBlackOrange and opc == '5'):
				pele = "Provalvemente tem a pele escura"
			elif(pixelBlackGreen > pixelBlackRed and pixelBlackGreen > pixelBlackOrange and opc == '6'):
				pele = "Provalvemente tem a pele muito escura"
			else:
				pele = "Erro! Seu tom de pele não condiz com a resposta"
			reshape = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))

			# rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
			# 	minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
			# # construct a list of bounding boxes from the detection
			# rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
			# update the data dictionary with the faces detected
			data.update({"opc": opc, "Color Skin": pele, "success": True})

			# return a JSON response
			return JsonResponse(data)
		except:
			data.update({"Erro":"Imagem não possui um rosto"})

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image