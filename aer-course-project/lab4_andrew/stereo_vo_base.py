"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0) # Blue
        self.inlier_color = (32,165,218) #Yellow

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor
    
    
    def compute_pose(self, prev_inliers, cur_inliers):
        centroid_prev_pose = np.mean(prev_inliers,axis=0)
        centroid_cur_pose = np.mean(cur_inliers,axis=0)

        #Construct Wba
        prev_inliers_origin = prev_inliers - centroid_prev_pose
        cur_inliers_origin = cur_inliers - centroid_cur_pose
        Wba = np.zeros((3,3))

        for (prev_pt,cur_pt) in zip(prev_inliers_origin, cur_inliers_origin):
            Wba += np.dot(cur_pt.reshape(3,-1),prev_pt.reshape(1,-1))
        
        #Utilize SVD
        USV_pose = np.linalg.svd(Wba)
        U = USV_pose[0]
        V_trans = USV_pose[2]

        #Account for reflection in rotation matrix
        arr = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,np.linalg.det(V_trans.T) * np.linalg.det(U)]
        ])
        
        #Compute Rotation Matrix
        Cba = U @ arr@ V_trans

        #Compute Translation
        translation = -Cba @ (-Cba.T @ centroid_cur_pose.reshape(3,-1) + centroid_prev_pose.reshape(3,-1))

        #Form rigid body transform
        T_ba = np.vstack(
            (np.hstack((Cba,translation)),
             np.array([0,0,0,1]))
        )

        return T_ba


    def ransac(self, pcl_prev, pcl_cur, num_iters=200, residual_threshold=5):
        max_inlier_count = 0
        max_inlier_idx = -1
        max_inlier_points =[]
        max_inlier_ids = []

        prev_point_cloud = pcl_prev
        cur_point_cloud = pcl_cur

        #Main for loop
        for iters in range(num_iters):

            #Sample points
            idx_drawn = np.random.default_rng().integers(0,len(cur_point_cloud),3)


            drawn_points_prev = np.vstack([
                prev_point_cloud[idx_drawn[0]].squeeze(),
                prev_point_cloud[idx_drawn[1]].squeeze(),
                prev_point_cloud[idx_drawn[2]].squeeze()
            ])

            drawn_points_cur = np.vstack([
                cur_point_cloud[idx_drawn[0]].squeeze(),
                cur_point_cloud[idx_drawn[1]].squeeze(),
                cur_point_cloud[idx_drawn[2]].squeeze()
            ])

            #Compute Centroid
            centroid_prev = np.mean(drawn_points_prev,axis=0)
            centroid_cur = np.mean(drawn_points_cur,axis=0)
        
            #Compute cross covariance matrix
            points_prev_origin = drawn_points_prev - centroid_prev
            points_cur_origin = drawn_points_cur - centroid_cur

            xcov =  points_prev_origin @ points_cur_origin

            #Utilize SVD
            USV = np.linalg.svd(xcov)

            #Compute Rotation Matrix
            V_transpose = USV[2]
            U = USV[0]

            arr = np.array([
                [1,0,0],
                [0,1,0],
                [0,0,np.linalg.det(V_transpose.T) * np.linalg.det(U)]
            ])

            C_ba = U @ arr @ V_transpose

            #Compute Translation
            translation = -C_ba @ (- C_ba.T @ centroid_cur.reshape(3,-1) + centroid_prev.reshape(3,-1))

            #Form rigid body transform
            T_ba = np.vstack((np.hstack((C_ba,translation)),np.array([0,0,0,1])))

            #Apply transformation to whole point cloud
            inlier_count = 0
            inlier_points = []
            inlier_ids = []

            for (idx,point) in enumerate(cur_point_cloud):

                #Convert to homogeneous coordinates
                temp_point = point.reshape(3,-1)
                temp_point = np.vstack((temp_point,1))

                predicted_point = (T_ba @ temp_point)[:3].squeeze()
                actual_point = prev_point_cloud[idx].squeeze()

                eucl_dist = np.linalg.norm(predicted_point - actual_point)

                if eucl_dist < residual_threshold:
                    inlier_points.append([actual_point,point.squeeze()])
                    inlier_count+=1
                    inlier_ids.append(idx)

            # Update inliers if better model is found
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                max_inlier_idx = iters
                max_inlier_points = inlier_points
                max_inlier_ids = inlier_ids

        return max_inlier_points, max_inlier_ids

    def pose_estimation(self, features_coor):
        # dummy C and r
        C = np.eye(3)
        r = np.array([0,0,0])
        # feature in right img (without filtering)
        f_r_prev, f_r_cur = features_coor[:,2:4], features_coor[:,6:8]
        # ------------- start your code here -------------- #
        
        '''
        Inputs: point clouds in both prev and cur frames, with correspondences between them
        outputs: filtered point clouds and a transformation between prev camera frame and cur camera frame
        Purpose: Translate inputs to outputs using RANSAC and ICP.

        '''
        #features_coor: idx0-2 is 

        #Convert to point cloud
        prev_point_cloud = []
        cur_point_cloud = []

        baseline = self.cam.baseline
        c_u = self.cam.cu
        c_v = self.cam.cv
        f_u = self.cam.fx
        f_v = self.cam.fy
        focal_length = self.cam.f_len

        for coor in features_coor:
            #Previous Frame
            u_l_prev = coor[0]
            u_r_prev = coor[2]
            v_l_prev = coor[1]
            v_r_prev = coor[3]
            
            scale = baseline/(u_l_prev - u_r_prev)

            prev_point = scale * np.array([

                [0.5 * (u_l_prev + u_r_prev) - c_u],

                [f_u / f_v * (0.5 * (v_l_prev + v_r_prev) - c_v)],

                [focal_length]
            ])

            prev_point_cloud.append(prev_point)

            #Current Frame
            u_l_cur = coor[4]
            u_r_cur = coor[6]
            v_l_cur = coor[5]
            v_r_cur = coor[7]

            scale = baseline/(u_l_cur - u_r_cur)

            cur_point = scale * np.array([

                [0.5 * (u_l_cur + u_r_cur) - c_u],

                [f_u / f_v * (0.5 * (v_l_cur + v_r_cur) - c_v)],

                [focal_length]
            ])

            cur_point_cloud.append(cur_point)


        #RANSAC filtering of the features
        if len(cur_point_cloud) < 3:
            print('Not enough points RANSAC.')
            return C, r, f_r_prev, f_r_cur

        else:

            max_inlier_points, max_inlier_ids = self.ransac(prev_point_cloud,cur_point_cloud,10,5)
            
        
        inlier_corrs = max_inlier_points #Inlier point correspondences [[prev,cur],...]

        if len(inlier_corrs) < 3:
            print('Not enough points ICP.')
            return C, r, f_r_prev, f_r_cur
        
        #Pose Estimation using ICP
        prev_points_inliers = np.vstack([pair[0] for pair in inlier_corrs])
        cur_points_inliers = np.vstack([pair[1] for pair in inlier_corrs])

        T_ba_icp = self.compute_pose(prev_points_inliers, cur_points_inliers)
        
        #Extract C and r
        C = T_ba_icp[:3,:3]
        r = T_ba_icp[:3,3]

        #Extract inliers
        f_r_prev = f_r_prev[max_inlier_ids]
        f_r_cur = f_r_cur[max_inlier_ids]

        # replace (1) the dummy C and r to the estimated C and r. 
        #         (2) the original features to the filtered features
        return C, r, f_r_prev, f_r_cur

    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 


