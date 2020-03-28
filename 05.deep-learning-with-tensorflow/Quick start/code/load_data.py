#!/usr/bin/python

## os.listdir(path) -- return the name of the file or folder.
## os.path.join("path_A/", "path_B") --combine two file directories

import os
from skimage import data
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_directory):
    # return the name the folders
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

def process_data():
    


ROOT_PATH = "/home/ubuntu16/deeplearning/dataset/"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training") 
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

images_array = np.array(images)
labels_array = np.array(labels)

# Print the `images` dimensions
print(images_array.ndim)
# Print the number of `images`'s elements
print(images_array.size)
# Print the first instance of `images`
images_array[0]

print(labels_array.ndim)
print(labels_array.size)
print(len(set(labels_array)))

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
# Show the plot
plt.show()

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))


# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    # Returns the index value at the beginning if the substring is included.
    image = images[labels.index(label)] 
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    # Returns the number of substrings that appear in a string.
    plt.title("Label {0} ({1})".format(label, labels.count(label))) 
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()