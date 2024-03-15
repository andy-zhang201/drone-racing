import cv2 as cv
import numpy as np
import os

img_path = "/home/andrewz/AER1217/student handout/output_folder"
dir_list = os.listdir(img_path) 
   
print("Files and directories in '", img_path, "' :")  
   
# print the list 
print(dir_list)

for i in range(len(dir_list)):
	cur_img_path = img_path + "/"+"image_" + str(i) + ".jpg"

	cur_img = cv.imread(cur_img_path) 
	print(cur_img_path)
	cv.imshow("img",cur_img)
	cv.waitKey(0)
	cv.destroyAllWindows() 