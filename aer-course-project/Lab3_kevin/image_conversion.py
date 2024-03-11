import numpy as np
import cv2 as cv
import os, os.path


def imageload(folder_path):
	images = []
	file_list = os.listdir(folder_path)
	for file in file_list:
		if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
			image = cv.imread(os.path.join(folder_path, file), cv.IMREAD_COLOR)
			if image is not None:
				images.append(image)
			else:
				print(f"Failed to load image: {file}")
	return images

def detect_circles(images):
	detected_circles = []
	for image in images:
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=500,
                                   param1=50, param2=30, minRadius=5, maxRadius=20)
		if circles is not None:
			detected_circles.append(np.uint16(np.around(circles[0, :])))
		else:
			detected_circles.append([])
	return detected_circles

def display(images, detected_circles):
	for idx, (image, circles) in enumerate(zip(images, detected_circles)):
		if len(circles)>0:
			for circle in circles:
				cv.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
				cv.imshow(f"Image {idx+1}", image)
				cv.waitKey(0)
				cv.destroyAllWindows()
		else:
			print(f"No circles detected in image {idx+1}")


if __name__ == "__main__":
	folder_path = "/home/vboxuser/drone-racing/aer-course-project/Lab3_kevin/img"
	images = imageload(folder_path)
	detected_circles = detect_circles(images)
	display(images,detected_circles)
	#print(test)
