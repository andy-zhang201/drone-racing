# Instructions to Run

Command to run:
```
python3 image_conversion_az.py
```

Ensure that the Python script file is in the same directory as the pose data file "lab3_pose.csv" and the folder containing your image data is named "img".

Note: After finding the transformation from the body to the world, we found that the quadrotor is flying mostly flat i.e. there is no inclination angle. This in turn led us to the assumption that the depth of the images is equal to pos_z of the quadrotor.
