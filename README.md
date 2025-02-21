# Image_feature_extraction
This code is about how to extract hand engineered features from a PV cell image
############################# the code is as below
from skimage.feature import greycomatrix, greycoprops  # Import functions for GLCM computation from scikit-image
# Function to convert float images to uint8
def float_to_uint8(image):
    return np.clip((image * 255.0), 0, 255).astype(np.uint8)  # Scale image to 0-255 and convert to uint8

# Function to compute Gray-Level Co-occurrence Matrix (GLCM) features for a given image
# greycomatrix() from Scikit-Image library, distance: defines pixel pair distance, angles: defines the orientation of pixel pairs, levels: the number of gray levels
def compute_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2], levels=256):
    glcm = greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)  # Compute GLCM
    features = {
        'contrast': greycoprops(glcm, 'contrast').ravel(),  # Extract contrast feature from GLCM
        # Additional features can be uncommented and used as needed
        # 'dissimilarity': greycoprops(glcm, 'dissimilarity').ravel(),
        # 'homogeneity': greycoprops(glcm, 'homogeneity').ravel(),
        # 'energy': greycoprops(glcm, 'energy').ravel(),
        # 'correlation': greycoprops(glcm, 'correlation').ravel(),
        # 'ASM': greycoprops(glcm, 'ASM').ravel()
    }
    return features  # Return the extracted features

# The shape of the _data array is (3780, 92, 92): _data is the array that stores our Pv cell image data in pickel format

# Convert all images to uint8
data_uint8 = [float_to_uint8(image) for image in _data]  # Apply the conversion function to each image in the dataset

# Compute GLCM features for each image
glcm_features = [compute_glcm_features(image) for image in data_uint8]  # Extract GLCM features for each image

# Convert GLCM features to a numpy array
X = np.array([list(features.values()) for features in glcm_features])  # Convert the list of feature dictionaries to a numpy array

# Reshape X to have shape (num_samples, num_features)
X_glcm = X.reshape(X.shape[0], -1)  # Reshape the array to have one row per image and one column per feature
