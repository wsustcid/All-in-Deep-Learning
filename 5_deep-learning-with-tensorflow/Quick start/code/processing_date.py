# Import the `transform` module from `skimage`
from skimage import transform 
import matplotlib.pyplot as plt
# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

from load_data import load_data


ROOT_PATH = "/home/ubuntu16/deeplearning/dataset/"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training") 
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]
# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)


traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    plt.title("shape: {0}, min: {1}, max: {2}".format(images28[traffic_signs[i]].shape, 
                                                  images28[traffic_signs[i]].min(), 
                                                  images28[traffic_signs[i]].max()))

    
# Show the plot
plt.show()

print(images28.shape)