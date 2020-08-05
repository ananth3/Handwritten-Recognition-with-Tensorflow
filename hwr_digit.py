import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

def hwr_digit(image):
	clf= joblib.load("/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/HDR/digits_cls.pkl")
	# im = cv2.imread(image)
	# h,w=image.shape[:2]
	# image=image[0:h,5:w]
	im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, im_th = cv2.threshold(im_gray, 125, 255, cv2.THRESH_BINARY_INV)
	kernel=np.ones((3,3),np.uint8)
	dilation=cv2.dilate(im_th,kernel,iterations=1)
	# Find contours in the image
	ctrs, hier = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	li=[]
	tup=[]
	numlist=[]
	for rect in rects:
		# Draw the rectangles
		cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
		# Resize the image
		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
		roi = cv2.dilate(roi, (3, 3))
		# Calculate the HOG features
		roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		# Predict digit
		nbr = clf.predict(np.array([roi_hog_fd],'float64'))
		# cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		li.append(rect[0])
		tuple1=(rect[0],int(nbr[0]))
		tup.append(tuple1)

	li.sort()

	for i in range(len(li)):
		n=[t for t in tup if t[0]==li[i]]
		numlist.append(str(n[0][1]))
	return "".join(numlist[1:])

	# 	print(n[0][1],end='')
	# print('\n')

def hwr_digits(image):
	clf= joblib.load("/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/HDR/digits_cls.pkl")
	image = cv2.imread('/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/HDR/di1.jpeg')
	im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)
	kernel=np.ones((3,3),np.uint8)
	dilation=cv2.dilate(im_th,kernel,iterations=1)
	# Find contours in the image
	ctrs, hier = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	li=[]
	tup=[]
	numlist=[]
	for rect in rects:
		# Draw the rectangles
		cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
		# Resize the image
		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
		roi = cv2.dilate(roi, (3, 3))
		# Calculate the HOG features
		roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		# Predict digit
		nbr = clf.predict(np.array([roi_hog_fd],'float64'))
		# cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		li.append(rect[0])

		tuple1=(rect[0],int(nbr[0]))
		tup.append(tuple1)

	li.sort()

	for i in range(len(li)):
		n=[t for t in tup if t[0]==li[i]]
		numlist.append(str(n[0][1]))
	return "".join(numlist)
	# 	print(n[0][1],end='')
	# print('\n')

