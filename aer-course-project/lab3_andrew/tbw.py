import numpy as np
import pdb
from image_conversion import imageload, detect_circles, display

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q0*q2 + 2*q1*q3],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q0*q1 + 2*q2*q3, 1 - 2*q1**2 - 2*q2**2]
    ])

def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)
    return data

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


def main():
    filename = 'lab3_pose.csv'  # Change this to the path of your CSV file
    data = read_csv_file(filename)

    #Find tf to body frame from world frame 
    transforms_dict = find_transforms(data)
    
    print(f'Transform at idx 5: \n {transforms_dict[5]}')

    folder_path = "images"
    img_dataset = imageload(folder_path)
    detected_circles = detect_circles(img_dataset)

    circle_centers = []
    for (idx, circle) in enumerate(detected_circles):
        if len(circle)>0:
            #append image idx and circle center
            circle_centers.append([idx,np.squeeze(circle)[0:2]])


    # breakpoint()
    # display(img_dataset,detected_circles)

    #For each circle center, find the depth
    for (idx, circle) in enumerate(circle_centers):

        #Find height of drone when taking this image from transformation matrix
        height = transforms_dict[circle[0]][2,3]
        z_val = transforms_dict[circle[0]][2,2]

        if z_val < 0.99:
            print(f'Low zval: {z_val}')

            
        #ASSUME GROUND IS FLAT
        #Perform Homograpy
        #Maybe we don't need to do it if 

        # breakpoint()


    #Find circle center coordinates in camera frame

    #Find circle center coordinates in world frame

if __name__ == "__main__":
    main()
