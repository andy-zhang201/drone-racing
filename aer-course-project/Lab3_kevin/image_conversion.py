import numpy as np
import cv2 as cv
import os, os.path
import re
import csv
import quaternion
import pandas as pd

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

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
    cords = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        circle_found = False
        for contour in contours:
            # Fit circle to contour
            (cx, cy), radius = cv.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            cords.append(center)
            radius = int(radius) 
            
            # Check circularity of the contour
            if cv.arcLength(contour, True) != 0:
                area = cv.contourArea(contour)
                perimeter = cv.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.25 and 30 <= radius <= 150:  # Adjust thresholds as needed
                    # Draw circle contour
                    cv.circle(image, center, radius, (0, 255, 0), 2)
                    cv.drawContours(image, [contour], 0, (255, 0, 0), 2)
                    print("Circle centroid coordinates:", center)
                    circle_found = True
        if circle_found:
            pass
        # 	cv.imshow('Detected Circle', image)
        # 	cv.waitKey(0)
        # 	cv.destroyAllWindows()
        else:
            center = (int(-1), int(-1))
            cords.append(center)
    return cords

# def detect_circles_old(images):
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

def dist_correct(img, camera_matrix, distortion_coefficients):
    undistorted_img = cv.undistort(img, camera_matrix, distortion_coefficients)
    return undistorted_img

def conv_CBI(xy_coordinates, camera_matrix, camera_to_body_transformation):
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    transformed_coordinates = []
    for (x, y) in xy_coordinates:
        point = np.array([[x], [y], [1]])
        point_camera_frame = np.dot(inv_camera_matrix, point)
        point_camera_frame_homogeneous = np.vstack((point_camera_frame, np.array([[1]])))
        point_body_frame_homogeneous = np.dot(camera_to_body_transformation, point_camera_frame_homogeneous)
        point_body_frame = point_body_frame_homogeneous[:-1] / point_body_frame_homogeneous[-1]
        point_body_frame = point_body_frame[:3]
        transformed_coordinates.append((point_body_frame[0][0], point_body_frame[1][0]))
    return transformed_coordinates

def conv_C_WB(point_body, drone_position, drone_orientation):
    drone_rotation_matrix = quaternion.as_rotation_matrix(drone_orientation)
    point_world_body_frame = np.dot(drone_rotation_matrix, np.hstack((point_body, np.ones((len(point_body), 1)))).T)
    point_world = point_world_body_frame[:3, :] + np.tile(drone_position.reshape(-1, 1), (1, len(point_body)))
    return point_world.T

if __name__ == "__main__":
	folder_path = "../lab3_andrew/images"
    file_path = "../lab3_andrew/lab3_pose.csv"
    drone_data = pd.read_csv(file_path)
    drone_position = drone_data[["p_x", "p_y", "p_z"]].values
    drone_orientation = drone_data[["q_w", "q_x", "q_y", "q_z"]].apply(lambda row: quaternion.quaternion(row["q_w"], row["q_x"], row["q_y"], row["q_z"]), axis=1)
    camera_matrix, distortion_coefficients = camera_param()
    images = imageload(folder_path, camera_matrix, distortion_coefficients)
    img_coordinates = detect_circles(images)
    T_CB = np.array([[0, -1, 0,0],
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    body_coordinates = conv_CBI(img_coordinates, camera_matrix, T_CB)
    for index, row in drone_data.iterrows():
            drone_pos = drone_position[index]
            drone_orient = drone_orientation[index]
            points_world = conv_C_WB(np.array(body_coordinates), drone_pos, drone_orient)
            print(f"Time step {index}: Points' positions in the world frame:")
            for point_world in points_world:
                print(point_world)


# 1 need to ignore all photos without circle, labeled as -1,-1 at the moment
# Should also ignore photoes that arent consecutively finding circle since there are still a few false positives but very little
# 2 average the world measured values to find the 6 image coordinates