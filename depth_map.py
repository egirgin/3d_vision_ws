import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import distance_transform_edt


def project_point_cloud_to_image(pcd, intrinsic_matrix, width, height, circle_radius=4):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    #plot_hist(points=points)
    colors = np.asarray(pcd.colors) * 255  # Convert color to 0-255 range

    # Initialize depth buffer and image
    depth_buffer = np.full((height, width), -np.inf)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a column of ones to convert to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    extrinsic_matrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0]
    ])
    
    projection_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)

    # Project the 3D points into the 2D image plane using the intrinsic matrix
    pixel_coords_homogeneous = projection_matrix @ points_homogeneous.T
    pixel_coords = pixel_coords_homogeneous[:2] / pixel_coords_homogeneous[2]
    depth_values = pixel_coords_homogeneous[2]

    #plot_hist(points=np.vstack((pixel_coords, depth_values)).T)

    # Round the pixel coordinates and cast to integers
    pixel_coords = np.round(pixel_coords).astype(int)

    # Iterate over the projected points
    for i, (x, y) in enumerate(pixel_coords.T):
        if 0 <= x < width and 0 <= y < height:
            depth = depth_values[i]
            if depth > depth_buffer[y, x]:
                depth_buffer[y, x] = depth
                #image[y, x] = colors[i]
                color = tuple(colors[i].astype(int).tolist())
                # Draw a circle on the image at the projected location
                cv2.circle(image, (x, y), circle_radius, color, thickness=-1)

    # Replace inf values with NaN for processing
    depth_buffer[depth_buffer == -np.inf] = np.nan

    # Fill missing depth values by propagating the nearest non-NaN value
    filled_depth_map = fill_missing_depth(depth_buffer)

    # Normalize the filled depth map for visualization
    normalized_depth_map = cv2.normalize(filled_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    colorful_depth_map = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_VIRIDIS)

    return image, colorful_depth_map

def fill_missing_depth(depth_map):
    # Create a binary mask where valid depths are 0 and invalid (NaN) are 1
    mask = np.isnan(depth_map).astype(np.uint8)

    # Compute the distance to the nearest valid depth for each pixel
    distance, indices = distance_transform_edt(mask, return_indices=True)

    # Use the indices to assign the nearest valid depth to NaN pixels
    filled_depth_map = depth_map[tuple(indices)]
    
    return filled_depth_map

def plot_hist(points):
    # Plot histograms for x, y, and z coordinates
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # X-coordinate histogram
    axes[0].hist(points[:, 0], bins=50, color='r', alpha=0.7)
    axes[0].set_title('X Coordinate Distribution')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Frequency')

    # Y-coordinate histogram
    axes[1].hist(points[:, 1], bins=50, color='g', alpha=0.7)
    axes[1].set_title('Y Coordinate Distribution')
    axes[1].set_xlabel('Y Position')
    axes[1].set_ylabel('Frequency')

    # Z-coordinate histogram
    axes[2].hist(points[:, 2], bins=50, color='b', alpha=0.7)
    axes[2].set_title('Z Coordinate Distribution')
    axes[2].set_xlabel('Z Position')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


from pathlib import Path

# Load the point cloud
sfm_path = Path("./logitech/sfm") # Path("example/sfm") # Path("custom_frames/sfm") # Path("./b950_office/sfm")

# Replace 'point_cloud_file' with the path to your .ply file
point_cloud_file = Path(sfm_path / "0/rec.ply")
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# translate and scale the point cloud if needed
# pcd.translate([0,0,0], True)
# pcd.scale(2500, np.array(pcd.get_center()))

# Define the desired resolution of the output image
width, height = 640, 480 

# Given intrinsic matrix
intrinsic_matrix = np.array([[830.32410584,   0.,         308.07273561],
                             [  0.,         830.85477646, 234.62985202],
                             [  0.,           0.,           1.        ]])
# Project the point cloud to the image plane
image, depth_map = project_point_cloud_to_image(pcd, intrinsic_matrix, width, height)

# Display the color image using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
orig_img = cv2.imread("./logitech/images/image_16.jpg")
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.title('Example Image')
plt.axis('off')  # Hide axis ticks

plt.subplot(1, 3, 2)
plt.imshow(image)
plt.title('Projected Point Cloud')
plt.axis('off')  # Hide axis ticks

# Display the depth map using matplotlib
plt.subplot(1, 3, 3)
plt.imshow(depth_map, cmap='gray')
plt.title('Depth Map')
plt.axis('off')  # Hide axis ticks

plt.show()
