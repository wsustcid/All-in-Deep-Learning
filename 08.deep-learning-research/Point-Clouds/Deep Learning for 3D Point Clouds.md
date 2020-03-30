<center><font size=7>  Deep Learning for 3D Point Clouds  </font>  </center> 
<img src=assets/Cover.jpg height=400>




<center> Shuai Wang </center>
<center>USTC, February 5, 2020


---

[TOC]

# Introduction

## 1. 3D Point Cloud Processing and Learning for Autonomous Driving

*Siheng Chen, Baoan Liu, Chen Feng, Carlos Vallespi-Gonzalez, Carl Wellington (Mitsubishi Electric Research Laboratories & Uber Advanced Technologies Group)*

### I. INTRODUCTION AND MOTIVATION

An autonomous system typically includes the sensing, map creation, localization, perception, prediction, routing, motion- planning, and control modules.

<img src=assets/1_1.png > 

A high- definition map is created offline. At runtime, the online system is given a destination. The system then senses its environment, localizes itself to the map, perceives the world around it and makes corresponding predictions of future motion for these objects. The motion planner uses these predictions to plan a safe trajectory for an autonomous vehicle (AV) to follow the route to the destination that is executed by the controller.



**3D point cloud processing:**

- 3D point cloud processing is the process of analyzing and modifying a 3D point cloud to optimize its transmission, storage and quality through various mathematical and computational algorithms. 
- Even though the processing algorithms could be significantly different, many processing tasks are naturally extended from 1D signal processing and 2D image processing. 
  - 3D point cloud compression is the 3D counterpart of image compression that aims to reduce the cost for storage or transmission of a 3D point cloud; 
  - 3D point cloud denoising is the 3D counterpart of image denoising that aims to remove noise from a 3D point cloud; 
  - 3D point cloud registration is the 3D counterpart of image registration that aims to align two or more 3D point clouds of the same scene; 
  - and 3D point cloud downsampling and upsampling are the 3D counterpart of image scaling that aims to change the resolution (number of points) in a 3D point cloud. 

**3D point cloud learning:**

- 3D point cloud learning is the process of interpreting and understanding a 3D point cloud. 
- With the powerful tools of deep neural networks, computer vision researchers aim to extend the success from images and videos to 3D point clouds. 
- Two primary learning problems are 3D point cloud recognition and segmentation. 
  - Similarly to the cases for 2D images, 3D point cloud recognition aims to classify a given 3D point cloud into a predefined class category 
  - and 3D point cloud segmentation aims to partition a given 3D point cloud into multiple segments. 
- Due to the irregular format of 3D point clouds, one of the biggest challenges for designing a learning algorithm is to formulate efficient data structures to represent 3D point clouds.  
  - Some algorithms transform 3D point clouds to regular 3D voxels, so that 3D convolutions can be used for the analysis; however, they have to make a trade-off between resolution and memory. 
  - To handle raw point clouds directly, PointNet [7] uses point-wise multilayer perceptrons (MLPs) and max-pooling to ensure the permutation invariance. After that, a series of 3D deep learning methods follow PointNet as their base networks.



### II. KEY INGREDIENTS OF 3D POINT CLOUD PROCESSING AND LEARNING

#### A. Properties

**Real-time LiDAR sweeps:**

- One real-time LiDAR sweep can naturally be organized on a 2D image, whose x-axis is the time stamp and y-axis is the laser ID. 
- We thus consider each individual real-time LiDAR sweep as an organized 3D point cloud. 
- For example, a Velodyne HDL-64E has 64 separate lasers and each laser fires thousands of times per second to capture a 360-degree field of view. We thus obtain a set of 3D points associated with 64 elevation angles and thousands of azimuth angles. (In a real-time LiDAR sweep, the vertical resolution is usually much lower than the horizontal resolution)
- Each collected 3D point is associated with a range measurement, an intensity value and a high precision GPS time stamps. 



#### B. Matrix representations

- raw points.  
- 3D voxelization.
- Range view. 
- Bird’s-eye view.

<img src=assets/1_2.png >

#### C. Representative tools

- Non-deep-learning methods. 
- Convolutional neural networks
- PointNet-based methods.
- Graph-based methods. ([11]-[16])

### III. 3D POINT CLOUD PROCESSING FOR HIGH-DEFINITION MAP CREATION

#### A. Overview of high-definition map creation module

<img src=assets/1_4.png />

Fig. 4: A standard HD map creation system includes two core components: 3D point cloud stitching and semantic feature extraction. 3D point cloud stitching usually adopts graph-based SLAM with hierarchical refinement; and semantic feature extraction contains iterative procedures of machine learning and human supervision. A key component in graph-based SLAM is a pose graph, modeling the relations among LiDAR poses. The nodes are LiDAR poses and edges reflecting the misalignment level between two LiDAR poses. The final outputs include a point-cloud map, which is a dense 3D point cloud, as well as a traffic-rule-related semantic feature map, containing the positions of landmarkers, traffic signs and traffic lights.

- Priors for localization. The
- Priors for perception
- Priors for prediction
- Priors for motion planning

### IV. 3D POINT CLOUD PROCESSING FOR LOCALIZATION

#### A. Overview of localization module

the localization module finds ego position of an autonomous vehicle relative to the reference position in the HD map. It consumes the real-time measure- ments from multiple sensors, including LiDAR, IMU, GPS, odometer, cameras, as well as the HD map; see Figure 5. 

<img src=assets/1_5.png >

Because of the 3D representation of an HD map, the ego position of an autonomous vehicle is a 6DOF pose (translation and rotation), which is a rigid transformation between the map frame and the LiDAR frame.

The importance of the localization module to autonomous driving is that it bridges the HD map to the other modules in an autonomy system. For example, by projecting the HD map priors, such as the lane geometries to the LiDAR frame, the autonomous vehicle gains the knowledge of which lane itself drives on and which lanes the detected traffics are on. 

High precision indicates the error of translation should be at the centimeter level and the error of rotation angle should be at the micro-radian level.

#### B. Map-based localization

**LiDAR-to-map registration.** The LiDAR-to-map registra-
tion component is to directly estimate the LiDAR pose by matching the LiDAR sweep to the the point-cloud map. Let

**Multisensor Fusion.** The multisensor fusion component is
to estimate a robust and confident pose from measurements of multiple sensors, including IMU, GPS, odometer, cameras, as well as the poses estimated by the LiDAR-to-map regis- tration module. The

### V. 3D POINT CLOUD PROCESSING FOR PERCEPTION

#### A. Overview of perception module 

The perception module is the visual system of an autonomous vehicle that enables the perception of the surrounding 3D environment. The input of the perception module usually includes the measurements from cameras, LiDAR, RADAR and ultrasound, as well as the ego- motion pose output from the localization module and the priors from the HD map. The outputs of the perception module are typically traffic light states and objects’ 3D bounding boxes with tracks.

Depending on the mechanism to fuse those modalities, a perception module can be categorized into late fusion and early fusion.

<img src=assets/1_6.png >

Each pipeline includes the detection component and the association and tracking component. The detection component finds bounding boxes and the association and tracking component tracks bounding boxes across frames to assign a unique identity for each individual object. A late- fusion module unifies the bounding box information from multiple pipelines and outputs a final 3D bounding-boxes with tracks.

To estimate traffic light states, a traffic light state estimator extracts the traffic light regions from images according to the position priors in an HD map and then it uses machine learning techniques to analyze the image and identify the traffic light state.

The industry has adopted the late-fusion-based approach for decades

A robust perception module usually includes multiple intermediate components, such as lane detection, 2D object detec- tion, 3D object detection, semantic segmentation and object tracking, to achieve the final goal. Among those components, 3D object detection is particularly interesting and challenging because it needs to handle real-time LiDAR sweeps and can directly produce the 3D bounding boxes for all objects in the scene. 

#### B. 3D object detection 

The task of 3D object detection is to detect and localize objects in the 3D space with the representation of bounding boxes based on one or multiple sensor measurements. 3D object detection usually outputs 3D bounding boxes of objects, which are the inputs for the component of object association and tracking. Based

**LiDAR-based detection**

Similarly to 2D objection detection, there are usually two paradigms of 3D object detection: single-stage detection and two-stage detection



<img src=assets/1_7.png >

To summarize, the input representation plays a crucial role in the LiDAR-based detection. The raw-point-based repre- sentation provides complete point information, but lacks the spatial prior. PointNet has become a standard method to handle this issue and extract features in the raw-point-based representation. The 3D voxelization-based representation and the BEV-based representation are simple and straightforward, but result in a lot of empty voxels and pixels. Feature pyramid networks with sparse convolutions can help address this issue. The range-view-based representation is more compact because the data is represented in the native frame of the sensor, leading to e.ficient processing, and it naturally models the occlusion. But objects at various ranges would have significantly differ- ent scales in the range-view-based representation, it usually requires more training data to achieve high performance. VFE introduces hybrid representations that take advantages of both the raw-point-based representations and the 3D voxelization- based representation. The one-stage detection tends to be faster and simpler, and naturally enjoys a high recall, while the two- stage detection tends to achieve higher precision [33], [23].

**Fusion-based detection.**

- A real-time LiDAR sweep provides a high-quality 3D representation of a scene; however, the measurements are generally sparse and only return in- stantaneous locations, making it difficult for LiDAR-based detection approaches to estimate objects’ velocities and detect small objects, such as pedestrians, at range. On the other hand, RADAR directly provides motion information and 2D images provides dense measurements. It is possible to naively merge detections from multiple modalities to improve overall robust- ness, but the benefit of this approach is limited. Following

- it remains an unresolved problem to design an effective early-fusion mechanism. The main challenges are the following: (1) measurements from each modality come from different measurement spaces. For example, 3D points are sparsely scattered in a continuous 3D space, while images contain dense measurements supported on a 2D lattice; (2) measurements from each modalty are not perfectly synchronized. LiDAR, camera and RADAR capture the scene at their own sampling frequencies; and (3) different sensing modalities have unique characteristics. The low-level processing of the sensor data depends on the individual sensor modality, but the high-level learning and fusion need to consider the characteristics across multiple modalities. Some

**Datasets.**

- KITTI [40] 
- nuScenes8, 
- Argoverse9, 
- Lyft Level 5 
- AV dataset10 
- and the Waymo open dataset11.

**Evaluation metrics.**

in academia are the precision- recall (PR) curve and average precision (AP); however, there is no standard platform to evaluate the running speed of each model. 

### APPENDIX

#### C. Elementary tasks

**1) 3D point cloud reconstruction:** The goal is to find a compact representation of a 3D point cloud that preserves the ability to reconstruct the original 3D point cloud; Reconstruction is helpful for data storage in au- tonomous driving.

<img src=assets/1_10.png width=400>



**2) 3D point cloud recognition:** The goal of recognition is to classify a 3D point cloud to a predefined category;

**3) 3D point cloud segmentation:** The goal of segmentation is to classify each 3D point in a 3D point cloud to a predefined category; see

**4) 3D point cloud denoising:** The goal of denoising is to remove noise from a noisy 3D point clouds and recover an original 3D point clouds;

**5) 3D point cloud downsampling:** The goal of downsampling is to select a subset of 3D points in an original 3D point cloud while peserving representative information; see

**6) 3D point cloud upsampling:** The goal of upsampling is to generate a dense (high-resolution) 3D point cloud from a sparse (low-resolution) 3D point cloud to describe the underlying geometry of an object or a scene. 3D point cloud upsampling is similar in nature to the super resolution of 2D images and is essentially an inverse procedure of downsam- pling; see

**7) 3D point cloud registration:** The goal of registration is to transform multiple 3D point clouds from local sensor frames into the standardized global frame. The key idea is to identify corresponding 3D points across frames and find a transformation that minimizes the distance (alignment error) between those correspondences. As a typical task of 3D point cloud processing, 3D point cloud registration is critical to the map creation module and the localization module of autonomous driving. In the map creation module, we need to register multiple LiDAR sweeps into the standardized global frame, obtaining a point-cloud map. In the localization module, we need to register a real-time LiDAR sweep to the point- cloud map to obtain the pose of the autonomous vehicle, which includes the position and the heading

## 2. Review: deep learning on 3D point clouds

### II. Challenges of deep learning on point clouds

<img src=assets/2_1.png >

These properties of point cloud are very challenging for deep learning, especially convolutional neural networks (CNN). These is because convolutional neural networks are based on convolution operation which is performed on a data that is ordered, regular and on a structured grid. Early approaches overcome these challenges by converting the point cloud into a structured grid format, section 3. However, recently re- searchers have been developing approaches that directly uses the power of deep learning on raw point cloud, see section 4, doing away with the need for conversion to structured grid.

### III. Structured grid based learning

The convolusion operation requires a struc- tured grid. Point cloud data on the other hand is unstructured, and this is a challenge for deep learning, and to overcome the challenge many approaches convert the point cloud data into a structured form. These approaches can be broadly divided into two categories, voxel based and multiview based. 

#### A. Voxel based

<img src=assets/2_3.png width=500>

#### B. multiview based

<img src=assets/2_4.png width=500>

#### C. Higher dimensional lattices

### IV. Deep learning directly on raw point cloud

#### A. PointNet

The design of PointNet, however, do not considers local dependency among points, thus, it does not cap- ture local structure. The global maxpooling applied select the feature vector in a ”winner–take –all” [26] principle, making it very susceptible to targetted adversarial attack as demonstrated in [27]. After PointNet many approaches were proposed to capture local structure.

#### B. Approaches with local structure computation

Many state-of-the-arts approaches where developed after Point- Net that captures local structure. These techniques capture local structure hierarchically in a smilar fashion to grid convolution with each heirachy encoding richer representation.
Basically, due to the inherent nature of point cloud of un- orderedness, local structure modeling rests on three basic op- erations: sampling; grouping; and a mapping function

<img src=assets/2_6.png width=400>

<img src=assets/2_7.png width=500 >

**4.2.1. Approaches that do not explore local correlation
Several**

Several approaches follow pointnet like MLP where correlation between points within a local region are not considered and in- stead, individual point features are learned via shared MLP and local region feature is aggregated using a maxpooling function in a winner-takes-all principle.

- PointNet++
- VoxelNet



**4.2.2 Approaches that explore local correlation**
Several approaches explore the local correlations between points in a local region to improve discriminative capabil- ity. This is intuitive because points do not exist in isolation, rather, multiple points together are needed to form a meaning- ful shape. PointCNN

- PointCNN

**4.2.3. Graph based**



### V. Benchmark Datasets

- Oxford Robotcar [74]. This dataset was developed by the Uni- versity of Oxford. It consists of around 100 times trajectories (a total of 101,046km trajectories) through central Oxford be- tween May 2014 to December 2015. This long-term dataset captures many challenging environment changes including sea- son, weather, traffic, and so on. And the dataset provides both images, LiDAR point cloud, GPS and INS ground truth for au- tonomous vehicles. The LIDRA data were collected by two SICK LMS-151 2D LiDAR scanners and one SICK LD-MRS 3D LIDAR scanner.
- NCLT[75]. It was developed by the University of Michigan. It contains 27 times trajectories through the University of Michi- gans North Campus between January 2012 to April 2013. This dataset also provides images, LiDAR, GPS and INS ground truth for long-term autonomous vehicles. The LiDRA point cloud was collected by a Velodyne-32 LiDAR scanner.
- DBNet [77]. This real-world LiDAR-Video dataset was devel- oped by Xiamen University et al. It aims at learning driving policy, since it is different from previous outdoor datasets. DB- Net provides LiDAR point cloud, video record, GPS and driver behaviors for driving behavior study. It contains 1,000 km driv- ing data captured by a Velodyne laser. NPM3D
- nuScenes [81]. The nuTonomy scenes (nuScenes) dataset pro- poses a novel metric for 3D object detection which was devel- oped by nuTonomy (an APTIV company). The metric consists of multi-aspects, which are classification, velocity, size, local- ization, orientation, and attribute estimation of the object. This dataset was acquired by an autonomous vehicle sensor suite (6 cameras, 5 radars and 1 lidar) with 360 degree field of view. It contains 1000 driving scenes collected from Boston and Singa- pore, where the two cities are both traffic-clogged. The objects in this dataset have 23 classes and 8 attributes, and they are all labeled with 3D bounding boxes.

### VI. Application of deep learning in 3D vision tasks

<img src=assets/2_8.png width=500 >

