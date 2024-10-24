import cv2
import numpy as np

def apply_box_blur(image, kernel_size=5):
	return cv2.blur(image, (kernel_size, kernel_size))

def apply_gaussian_blur(image, kernel_size=(15, 15)): #nenhum valor no kernel_size pode ser par
	return cv2.GaussianBlur(image, kernel_size, 0)

def apply_gaussian_blur_bilinear(image, kernel_size=(0.5, 0.5)):
	kernel_size = [int(kernel_size[0] * image.shape[0]), int(kernel_size[1] * image.shape[1])]
	for i in range(2):
		if kernel_size[i] % 2 == 0:
			kernel_size[i] += 1
	#print(kernel_size)
	return apply_gaussian_blur(image, kernel_size)

def apply_radial_blur(image, num_rotations=10, blur_strength=5):
    
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Define the center of the image (center of the radial blur)
    center_x, center_y = width // 2, height // 2
    
    # Create an empty array to accumulate the blurred image
    blurred_image = np.zeros_like(image, dtype=np.float32)
    
    # Perform multiple rotations and blend them
    for i in range(num_rotations):
        # Calculate rotation angle for each step
        angle = blur_strength * i
        
        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Rotate the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Accumulate the rotated image
        blurred_image += rotated_image.astype(np.float32)
    
    # Average the accumulated images
    blurred_image /= num_rotations
    
    # Convert the result back to an 8-bit image
    return cv2.convertScaleAbs(blurred_image)


def apply_directional_blur(image, kernel_size=15, angle=0):
    
    # Create a blank kernel with the specified kernel size
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Calculate the direction of the blur
    center = kernel_size // 2
    radians = np.deg2rad(angle)
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)
    
    for i in range(kernel_size):
        x = int(center + (i - center) * cos_angle)
        y = int(center + (i - center) * sin_angle)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Normalize the kernel (so the sum of its values is 1)
    kernel /= kernel.sum()
    
    # Apply the directional blur using filter2D
    return cv2.filter2D(image, -1, kernel)


def apply_zoom_radial_blur(image, num_scales=20, zoom_factor=1.02):
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create an empty float32 array to accumulate the scaled images
    accumulated_image = np.zeros_like(image, dtype=np.float32)
    
    # Perform the zooming by scaling the image multiple times
    for i in range(num_scales):
        # Scale the image by progressively smaller amounts
        scale = zoom_factor ** i
        
        # Calculate the new dimensions after scaling
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        
        # Resize the image to the new dimensions
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        
        # Crop the scaled image around the center of the original image
        x_start = (scaled_width - width) // 2
        y_start = (scaled_height - height) // 2
        cropped_image = scaled_image[y_start:y_start+height, x_start:x_start+width]
        
        # Accumulate the cropped images
        accumulated_image += cropped_image.astype(np.float32)
    
    # Average the accumulated images
    accumulated_image /= num_scales
    
    # Convert the result back to an 8-bit image
    return cv2.convertScaleAbs(accumulated_image)



def apply_variable_blur(image, focal_point, max_blur_radius=30):

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a blurred image to accumulate results
    blurred_image = np.zeros_like(image, dtype=np.float32)

    # Create a mask for variable blur
    for y in range(height):
        for x in range(width):
            # Calculate the distance from the focal point
            distance = np.sqrt((x - focal_point[0]) ** 2 + (y - focal_point[1]) ** 2)
            # Determine the blur radius based on distance
            blur_radius = min(max_blur_radius, distance / 5)  # Adjust the factor as needed

            if blur_radius > 0:
                # Create a kernel for blur
                kernel_size = int(blur_radius) | 1  # Ensure kernel size is odd
                # Apply Gaussian blur with the calculated radius
                blurred_section = cv2.GaussianBlur(image[y:y+1, x:x+1], (kernel_size, kernel_size), blur_radius)
                blurred_image[y, x] = blurred_section

            else:
                blurred_image[y, x] = image[y, x]

    # Convert the resulting image back to uint8
    return cv2.convertScaleAbs(blurred_image)






"""# Example of using the function
image_path = "example_image.jpg"

# Load the image from the given path
image = cv2.imread(image_path)
    
# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load image from {image_path}")

#img = apply_gaussian_blur(image, kernel_size=(15, 301))
img = apply_gaussian_blur_bilinear(image, kernel_size=(0, .5))
#img = apply_radial_blur(image)
#img = apply_directional_blur(image, kernel_size=15, angle=45)
#img = apply_zoom_radial_blur(image, num_scales=30, zoom_factor=1.03)
#img = apply_box_blur(image, kernel_size=10)
#img = apply_variable_blur(image, focal_point=(250, 250))

#def iscollection(obj): return isinstance(obj, list) or isinstanec(obj, tuple)

#display_resize_factor = 1.0 #0.25
resized = cv2.resize(img, (int(img.shape[1] * display_resize_factor), int(img.shape[0] * display_resize_factor)))

if resized is not None:
    cv2.imshow('Blurred Image (Resized)', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
