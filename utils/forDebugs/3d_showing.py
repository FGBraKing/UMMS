import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict
from data.transforms.transformOnArray import normalize, NormalizeRange
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3D(img, threshold=-400):
    verts, faces = measure.marching_cubes(img, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, img.shape[0])
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[2])

    plt.show()


# def generate_mesh(image, isovalue=0, channel=0):
#     """
#     Creates and returns a Mesh object
#     :param image: an AICSImage object
#     :param isovalue: The value that is used to pick the isosurface returned by the marching cubes algorithm
#                      For more info: https://www.youtube.com/watch?v=5fNbCFjqWao @ 40:00 mins
#     :param channel: The channel in the image that is used to extract the isosurface
#     :return: A Mesh object
#     """
#     if not isinstance(image, AICSImage):
#         raise ValueError("Meshes can only be generated with AICSImage objects!")
#     if channel >= image.size_c:
#         raise IndexError("Channel provided for mesh generation is out of bounds for image data!")
#     image_stack = image.get_image_data("ZYX", C=channel)
#     # Use marching cubes to obtain the surface mesh of the membrane wall
#     verts, faces, normals, values = measure.marching_cubes(image_stack, isovalue, allow_degenerate=False)
#     return Mesh(verts, faces, normals, values)
#
#
# def marching_cubes(field,iso=0.5):
#     try:
#         from skimage.measure import marching_cubes
#         surface_points, surface_triangles = marching_cubes(density_field,iso)
#
#     except ImportError:
#         print "Please try to install SciKit-Image!"
#
#         from mayavi import mlab
#         from mayavi.mlab import contour3d
#
#         mlab.clf()
#         surface = mlab.contour3d(field,contours=[iso])
#
#         my_actor=surface.actor.actors[0]
#         poly_data_object=my_actor.mapper.input
#         surface_points = (np.array(poly_data_object.points) - np.array([abs(grid_points/2.),abs(grid_points/2.),abs(grid_points/2.)])[np.newaxis,:])*(grid_max/abs(grid_points/2.))
#         surface_triangles = poly_data_object.polys.data.to_array().reshape([-1,4])
#         surface_triangles = surface_triangles[:,1:]
#
#     return surface_points, surface_triangles


def plot_3d_cubic(image):
    '''
        plot the 3D cubic
    :param image:   image saved as npy file path
    :return:
    '''
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = np.load(image)
    verts, faces = measure.marching_cubes(image,0)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


def visualize_voxel_spectral(points, vis_size=128):
    """Function to visualize voxel (spectral)."""
    points = np.rint(points)
    points = np.swapaxes(points, 0, 2)
    fig = plt.figure(figsize=(1, 1), dpi=vis_size)
    verts, faces, _, _ = measure.marching_cubes(points, 0, spacing=(0.1, 0.1, 0.1))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral_r', lw=0.1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(
        fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(
        vis_size, vis_size, 3)
    plt.close('all')
    return data






