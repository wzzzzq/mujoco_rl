"""Utilities to load example Piper robot model."""

import coal
import numpy as np
import os
import pinocchio
from plyfile import PlyData

from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models(use_sphere_collisions=False):
    """
    Gets the example Piper models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "piper_description")
    urdf_filename = "piper.urdf"
    # urdf_filename = "piper_spheres.urdf" if use_sphere_collisions else "piper.urdf"
    urdf_filepath = os.path.join(package_dir, "urdf", urdf_filename)

    return pinocchio.buildModelsFromUrdf(urdf_filepath, package_dirs=models_folder)


def load_point_cloud(pointcloud_path=None, voxel_resolution=0.04):
    """
    Loads a point cloud from a PLY file and converts it into an octree structure.

    Returns
    -------
    octree : coal.Octree
        An octree data structure representing the hierarchical spatial partitioning
        of the point cloud. The voxel resolution default value is set to 0.04 units.
    """
    if pointcloud_path is None:
        models_folder = get_example_models_folder()
        pointcloud_path = os.path.join(
            models_folder, "example_point_cloud", "example_point_cloud.ply"
        )

    ply_data = PlyData.read(pointcloud_path)
    vertices = ply_data["vertex"]
    vertex_array = np.array([vertices["x"], vertices["y"], vertices["z"]]).T
    octree = coal.makeOctree(vertex_array, voxel_resolution)

    return octree


def add_self_collisions(model, collision_model, srdf_filename=None):
    """
    Adds link self-collisions to the Piper collision model.

    Uses an SRDF file to remove excluded collision pairs.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Piper model.
        collision_model : `pinocchio.Model`
            The Piper collision geometry model.
        srdf_filename : str, optional
            Path to the SRDF file describing the excluded collision pairs.
            Defaults to the one included with Piper model.
    """
    if srdf_filename is None:
        models_folder = get_example_models_folder()
        package_dir = os.path.join(models_folder, "piper_description")
        srdf_filename = os.path.join(package_dir, "srdf", "piper.srdf")
        print(f"srdf_filename : {srdf_filename}")

    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)


def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    """
    Adds obstacles and collisions to the Piper collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Piper model.
        collision_model : `pinocchio.Model`
            The Piper collision geometry model.
        visual_model : `pinocchio.Model`
            The Piper visual geometry model.
        inflation_radius : float, optional
            Inflation radius in meters around objects.
    """
    # Example ground plane (adjust height and size if needed)
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),
        coal.Box(3.0, 3.0, 0.01),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    # Example obstacles (positions and sizes may need tuning for Piper workspace)
    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.2, 0.0, 1.0])),
        coal.Sphere(0.25 + inflation_radius),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.4, 0.7])),
        coal.Box(
            0.3 + 2.0 * inflation_radius,
            0.5 + 2.0 * inflation_radius,
            0.5 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    # Activate collisions between Piper links and obstacles
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "piper" in cobj.name
    ]
    obstacle_names = ["ground_plane", "obstacle_sphere_1", "obstacle_box_1"]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)

    # Disable collision between Piper base and ground plane
    set_collisions(model, collision_model, "base_link", "ground_plane", False)


def add_octree_collisions(model, collision_model, visual_model, octree):
    """
    Adds an octree collision/visual object to the Piper model and enables collisions
    between octree and Piper links.

    Parameters
    ----------
    model : `pinocchio.Model`
        The Piper model.
    collision_model : `pinocchio.Model`
        The Piper collision geometry model.
    visual_model : `pinocchio.Model`
        The Piper visual geometry model.
    octree : coal.Octree
        Octree representing environment obstacles.
    """
    octree_object = pinocchio.GeometryObject(
        "octree", 0, pinocchio.SE3.Identity(), octree
    )
    octree_object.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    collision_model.addGeometryObject(octree_object)
    visual_model.addGeometryObject(octree_object)

    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "piper" in cobj.name
    ]

    for collision_name in collision_names:
        set_collisions(model, collision_model, "octree", collision_name, True)
