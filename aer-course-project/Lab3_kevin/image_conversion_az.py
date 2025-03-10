import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os, os.path
import re
import csv
import quaternion
import pandas as pd
from sklearn.cluster import KMeans

def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)
    return data

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q0*q2 + 2*q1*q3],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q0*q1 + 2*q2*q3, 1 - 2*q1**2 - 2*q2**2]
    ])

def find_transforms(data):
    transforms_lookup = {}
    for line in data[1:]: #Read starting from second line
        #Convert each string element in line to a float and save in variables
        img_id, x, y, z, qw, qx, qy, qz = map(float, line)
        
        # XYZ position vector
        position = np.array([x, y, z])
        
        # XYZW quaternion
        quaternion = np.array([qw, qx, qy, qz])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # Create homogeneous transformation matrix Tbw
        Tbw = np.eye(4)
        Tbw[:3, :3] = rotation_matrix
        Tbw[:3, 3] = position
        
        transforms_lookup[img_id] = Tbw

    return transforms_lookup

def camera_param():
    focal_length_x = 698.86
    principal_point_x = 306.91
    focal_length_y = 699.13
    principal_point_y = 150.34
    k1 = 0.191887
    k2 = -0.56368
    p1 = -0.003676
    p2 = -0.002037
    k3 = 0
    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
                            [0, focal_length_y, principal_point_y],
                            [0, 0, 1]])
    distortion_coefficients = np.array([k1, k2, p1, p2, k3])
    return camera_matrix, distortion_coefficients
    
def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def dist_correct(img, camera_matrix, distortion_coefficients):
    undistorted_img = cv.undistort(img, camera_matrix, distortion_coefficients)
    return undistorted_img

def imageload(folder_path, camera_matrix, distortion_coefficients):
    images = []
    file_list = sorted_nicely(os.listdir(folder_path))
    for file in file_list:
        if file.lower().endswith('.jpg'):
            image = cv.imread(os.path.join(folder_path, file))
            image = dist_correct(image, camera_matrix, distortion_coefficients)
            images.append(image)
    return images

def detect_circles(images):
    '''
    Returns a DICTIONARY {img_idx : [(x1,y1),(x2,y2),...]}
    '''

    cords = {}

    for (img_idx, image) in enumerate(images):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        circle_found = False
        for contour in contours:
            # Fit circle to contour
            (cx, cy), radius = cv.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            
            # Check circularity of the contour
            if cv.arcLength(contour, True) != 0:
                area = cv.contourArea(contour)
                perimeter = cv.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.25 and 30 <= radius <= 150:  # Adjust thresholds as needed
                    # Draw circle contour
                    cv.circle(image, center, int(radius), (0, 255, 0), 2)
                    cv.drawContours(image, [contour], 0, (255, 0, 0), 2)
                    # print("Circle centroid coordinates:", center)
                    
                    #Check if a circle has already been found:
                    if img_idx in cords:
                        cords[img_idx].append(center)

                    else:
                        cords[img_idx] = [center] #Create list of tuples representing detected circle centers

                    circle_found = True

    return cords

# def detect_circles_old(images): # Different method of calculation with HoughCircles
# 	detected_circles = []
# 	for image in images:
# 		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=500,
#                                    param1=50, param2=30, minRadius=5, maxRadius=20)
# 		if circles is not None:
# 			detected_circles.append(np.uint16(np.around(circles[0, :])))
# 		else:
# 			detected_circles.append([])
# 	return detected_circles

def find_body_coords_from_px(center_pixels, camera_matrix, T_CB, drone_height):
    '''
    Convert list of tuples of pix coords to list of numpy arrays representing body frame coordinates
    Don't need to undistort as image is undistorted when we load the images
    '''
    body_points = []
    inv_camera_matrix = np.linalg.inv(camera_matrix)

    for center_point in center_pixels:
        point_camera_frame_normalized = np.matmul(inv_camera_matrix, np.array([[center_point[0]],[center_point[1]],[1]]))
        point_camera_frame = drone_height*point_camera_frame_normalized

        #Rotate to body frame
        T_BC = np.linalg.inv(T_CB)

        point_body_frame_homogeneous = np.matmul(T_BC, np.vstack([point_camera_frame, np.array([[1]])]))
        point_body_frame = point_body_frame_homogeneous[:3]
        body_points.append(point_body_frame)
    return body_points

def find_world_coords_from_body(points_body, T_WB):
    '''
    Replace center_pixels (list of tuples) with world_pixels (list of numpy arrays)
    '''

    #Construct rigid body tf
    points_world_list = []

    for body_pt in points_body:

        #Transform to world coords
        world_pt = np.matmul(T_WB, np.vstack([body_pt.reshape(3,-1),[1]]))
        points_world_list.append(world_pt[:3])

    #Return list of numpy arrays
    return points_world_list

if __name__ == "__main__":

    #Define File paths
    file_path = "img"
    filename = "lab3_pose.csv" # Change this to the path of your CSV file
    
    #Find rigid body transforms of drone in world frame 
    data = read_csv_file(filename)
    transforms_dict = find_transforms(data)

    #Load camera parameters
    camera_matrix, distortion_coefficients = camera_param()

    #Load images in sorted order and correct distortion
    images = imageload(file_path, camera_matrix, distortion_coefficients)
    
    #Detect Circles in images. Return a dictionary of lists containing img_idx:[center_x,center_y].
    #If no detected circles, img_idx will not be present
    img_coordinates = detect_circles(images)

    #Load camera to body transform
    T_CB = np.array([[0, -1, 0,0],
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])


    #For each image index where you found circles, find the world coordinates of their centers
    
    points_world = {}
    for index, center_pixels in img_coordinates.items():

	    # Drone height can be directly used for image depth given the minimal angle change from steady level hover
        drone_height = transforms_dict[index][2,3]

        #Find circle centers in body frame coordinates
        points_body = find_body_coords_from_px(center_pixels, camera_matrix, T_CB, drone_height) 

        #Find circle centers in world frame coordinates
        detected_points_world = find_world_coords_from_body(points_body, transforms_dict[index])
        
        #Append to dictionary
        if (index in points_world):
            points_world[index].append(detected_points_world)

        else:
            world_coords_array = detected_points_world
            points_world[index] = world_coords_array

    #Flatten the list of numpy arrays into a single numpy array
    samples=np.array([])
    for idx, points in points_world.items():
        for pt in points:
            #Apply K-means clustering to identify landmark locations

            if samples.size ==0:
                samples = np.array(pt).reshape(-1)

            else:
                samples = np.vstack([samples,np.array(pt).reshape(-1)])
    

    kmeans = KMeans(n_clusters=6, random_state=0).fit(samples)

    # The centroids of the clusters are the accurately localized positions of the landmarks
    landmark_positions = kmeans.cluster_centers_
    x = landmark_positions[:,0]
    y = landmark_positions[:,1]

    x_samples = samples[:,0]
    y_samples = samples[:,1]

    print("Accurately localized positions of the landmarks:")
    print(landmark_positions)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_samples, y_samples, c='blue',alpha=0.5)  # s is size, c is color
    plt.scatter(x, y, c='red')  # s is size, c is color
    plt.title('Landmark Locations')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()
