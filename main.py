import shutil, sys, os
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging
from pycolmap import ImageReaderOptions

def resize_imgs(input_path, output_path, width, height):
    output_path.mkdir(exist_ok=True)
    import cv2
    for img_filename in os.listdir(input_path):
        image = cv2.imread(input_path / img_filename)
        resized_image = cv2.resize(image, (width, height))  # Set desired width and height
        cv2.imwrite(output_path / img_filename, resized_image)

def run():
    output_path = Path("logitech") # Path("custom_frames") # Path("example/") # Path("b950_office")
    image_path = output_path / "images" #"openairlab/images" # "Fountain/images" # images
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"
    #new_image_path = output_path / "Fountain/resized_images"

    #resize_imgs(image_path, new_image_path, width=1920, height=1080)

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")  # + time

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    if not image_path.exists():
        logging.info("Downloading the data.")
        zip_path = output_path / "data.zip"
        urllib.request.urlretrieve(data_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as fid:
            fid.extractall(output_path)
        logging.info(f"Data extracted to {output_path}.")

    if database_path.exists():
        database_path.unlink()

    import time
    start = time.time()
    pycolmap.extract_features(database_path, image_path) #new_image_path)
    pycolmap.match_exhaustive(database_path)
    num_images = pycolmap.Database(database_path).num_images

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    with enlighten.Manager() as manager:
        with manager.counter(total=num_images, desc="Images registered:") as pbar:
            pbar.update(0, force=True)
            recs = pycolmap.incremental_mapping(
                database_path,
                image_path, # image_path new_image_path
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    print("Took {} secs".format(time.time()-start))
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

    reconstruction = pycolmap.Reconstruction(sfm_path / "0")

    reconstruction.export_PLY(sfm_path / "0/rec.ply")  # PLY format

    visualize(sfm_path=sfm_path)

def visualize(sfm_path):
    import open3d as o3d

    # Replace 'point_cloud_file' with the path to your .ply file
    point_cloud_file = Path(sfm_path / "0/rec.ply")
    pcd = o3d.io.read_point_cloud(str(point_cloud_file))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.translate([0, 0, 1])
    mesh_sphere.paint_uniform_color([0, 0, 0])

    o3d.visualization.draw_geometries([pcd, mesh_sphere, axes])

if __name__ == "__main__":
    run()