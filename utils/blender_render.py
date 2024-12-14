import pickle
import numpy as np
import os
import sys
import bpy
import math
import shutil
import json
import time
from mathutils import Vector, Matrix
import argparse
import glob
import colorsys
import bmesh
from mathutils.bvhtree import BVHTree


context = bpy.context
scene = context.scene
render = scene.render


#########################################################
# Blender scene setup
#########################################################
def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def setup_blender_env(img_width, img_height):

    reset_scene()

    # Set render engine and parameters
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = img_width
    render.resolution_y = img_height
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 64
    scene.cycles.samples = 64  # 32 for testing, 256 or higher 512 for final
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    preferences = context.preferences
    preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    preferences.addons["cycles"].preferences.get_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])


#########################################################
# Blender camera setup
#########################################################
def create_camera_list(c2w, K):
    """
    Create a list of camera parameters

    Args:
        c2w: (N, 4, 4) camera to world transform
        K: (3, 3) or (N, 3, 3) camera intrinsic matrix
    """
    cam_list = []
    for i in range(len(c2w)):
        pose = c2w[i].reshape(-1, 4)
        if len(K.shape) == 3:
            cam_list.append({'c2w': pose, 'K': K[i]})
        else:
            cam_list.append({'c2w': pose, 'K': K})
    return cam_list


def setup_camera():
    # Find a camera in the scene
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            print("found camera")
            break
    # If no camera is found, create a new one
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    # Set the camera as the active camera for the scene
    bpy.context.scene.camera = cam
    return cam


class Camera():
    def __init__(self, im_height, im_width, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.w = im_width
        self.h = im_height
        self.camera = setup_camera()
        
    def set_camera(self, K, c2w):
        self.K = K       # (3, 3)
        self.c2w = c2w   # (3 or 4, 4), camera to world transform
        # original camera model: x: right, y: down, z: forward (OpenCV, COLMAP format)
        # Blender camera model:  x: right, y: up  , z: backward (OpenGL, NeRF format)
        
        self.camera.data.type = 'PERSP'
        self.camera.data.lens_unit = 'FOV'
        f = K[0, 0]
        rad = 2 * np.arctan(self.w/(2 * f))
        self.camera.data.angle = rad
        self.camera.data.sensor_fit = 'HORIZONTAL'  # 'HORIZONTAL' keeps horizontal right (more recommended)

        # f = K[1, 1]
        # rad = 2 * np.arctan(self.h/(2 * f))
        # self.camera.data.angle = rad
        # self.camera.data.sensor_fit = 'VERTICAL'  # 'VERTICAL' keeps vertical right
        
        self.pose = self.transform_pose(c2w)
        self.camera.matrix_world = Matrix(self.pose)
        
    def transform_pose(self, pose):
        '''
        Transform camera-to-world matrix
        Input:  (3 or 4, 4) x: right, y: down, z: forward
        Output: (4, 4)      x: right, y: up  , z: backward
        '''
        pose_bl = np.zeros((4, 4))
        pose_bl[3, 3] = 1
        # camera position remain the same
        pose_bl[:3, 3] = pose[:3, 3] 
        
        R_c2w = pose[:3, :3]
        transform = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ]) 
        R_c2w_bl = R_c2w @ transform
        pose_bl[:3, :3] = R_c2w_bl
        
        return pose_bl

    def initialize_depth_extractor(self):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.view_layer.cycles.denoising_store_passes = True
        bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        render_layers = nodes['Render Layers']
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.name = 'File Output Depth'
        depth_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs[2], depth_file_output.inputs[0])

    def render_single_timestep_rgb_and_depth(self, cam_info, FRAME_INDEX, dir_name_rgb='rgb', dir_name_depth='depth'):

        dir_path_rgb = os.path.join(self.out_dir, dir_name_rgb)
        dir_path_depth = os.path.join(self.out_dir, dir_name_depth)
        os.makedirs(dir_path_rgb, exist_ok=True)
        os.makedirs(dir_path_depth, exist_ok=True)

        self.set_camera(cam_info['K'], cam_info['c2w'])

        # Set paths for both RGB and depth outputs
        depth_output_path = os.path.join(dir_path_depth, '{:0>3d}'.format(FRAME_INDEX))
        rgb_output_path = os.path.join(dir_path_rgb, '{:0>3d}.png'.format(FRAME_INDEX))

        # Assuming your Blender setup has nodes named accordingly
        bpy.context.scene.render.filepath = rgb_output_path
        bpy.data.scenes["Scene"].node_tree.nodes["File Output Depth"].base_path = depth_output_path

        bpy.ops.render.render(use_viewport=True, write_still=True)


#########################################################
# Object manipulation
#########################################################
def object_meshes(single_obj):
    for obj in [single_obj] + single_obj.children_recursive:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bpy.ops.object.select_all(action="DESELECT")
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else object_meshes(single_obj):
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene(single_obj):
    bbox_min, bbox_max = scene_bbox(single_obj)
    scale = 1 / max(bbox_max - bbox_min)
    single_obj.scale = single_obj.scale * scale
    bpy.context.view_layer.update()             # Ensure the scene is fully updated
    bbox_min, bbox_max = scene_bbox(single_obj)
    offset = -(bbox_min + bbox_max) / 2
    single_obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def load_object(object_path: str) -> bpy.types.Object:
    """Loads an object asset into the scene."""
    # import the object
    if object_path.endswith(".glb") or object_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path, axis_forward='Y', axis_up='Z')
    elif object_path.endswith(".ply"):
        # bpy.ops.import_mesh.ply(filepath=object_path)                             # only used for snap blender
        bpy.ops.wm.ply_import(filepath=object_path, forward_axis='Y', up_axis='Z')  # used for blender 4.0 & snap blender
    elif object_path.endswith(".obj"):
        # bpy.ops.import_scene.obj(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')  # only used for snap blender
        bpy.ops.wm.obj_import(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')       # used for blender 4.0 & snap blender
    ##### This part is used for ChatSim assets #####
    elif object_path.endswith(".blend"):
        blend_path = object_path
        new_obj_name = 'chatsim_' + blend_path.split('/')[-1].split('.')[0]
        model_obj_name = 'Car'                                                  # general names used for all assets in ChatSim
        with bpy.data.libraries.load(blend_path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:                                             # actually part that import the object
            if obj.name == model_obj_name:
                bpy.context.collection.objects.link(obj)
        if model_obj_name in bpy.data.objects:                                  # rename the object to avoid conflict
            imported_object = bpy.data.objects[model_obj_name]
            imported_object.name = new_obj_name
            print(f"rename {model_obj_name} to {new_obj_name}")
        for slot in imported_object.material_slots:                             # rename the material to avoid conflict
            material = slot.material
            if material:
                material.name = new_obj_name + "_" + material.name
        return imported_object
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    new_obj = bpy.context.object
    return new_obj


def merge_meshes(obj):
    """
    Merge all meshes within the object into a single mesh

    Args:
        obj: blender object
    """
    all_object_nodes = [obj] + obj.children_recursive
    mesh_objects = [obj for obj in all_object_nodes if obj.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')     # Deselect all objects first
    for obj in mesh_objects:
        obj.select_set(True)                         # Select each mesh object
        bpy.context.view_layer.objects.active = obj  # Set as active object
    bpy.ops.object.join()
    bpy.ops.object.select_all(action="DESELECT")


def remove_empty_nodes_v1(obj):
    """
    Remove empty nodes in the object (original version)

    Args:
        obj: blender object
    
    Returns:
        obj: blender object with empty nodes removed, only keep the mesh nodes
    """
    all_object_nodes = [obj] + obj.children_recursive
    
    # find the mesh node first
    current_obj = None
    world_matrix = None
    for obj_node in all_object_nodes:
        if obj_node.type == 'MESH':
            current_obj = obj_node      # after merging, only one mesh node left
            world_matrix = obj_node.matrix_world.copy()
            break
    
    # perform removal    
    for obj_node in all_object_nodes:
        if obj_node != current_obj:
            bpy.data.objects.remove(obj_node, do_unlink=True)
            
    # apply world transform back
    current_obj.matrix_world = world_matrix
            
    bpy.ops.object.select_all(action="DESELECT")
    return current_obj


def remove_empty_nodes_v2(obj):
    """
    Remove empty nodes in the object while preserving mesh transformations.
    
    Args:
        obj: Blender object, typically the root of a hierarchy.
    
    Returns:
        Blender object with empty nodes removed, only keeping the mesh nodes.
    """
    all_object_nodes = [obj] + obj.children_recursive
    mesh_node = None

    # Find the first mesh node
    for obj_node in all_object_nodes:
        if obj_node.type == 'MESH':
            mesh_node = obj_node
            break

    if mesh_node:
        # Clear parent to apply transformation, if any
        mesh_node.matrix_world = mesh_node.matrix_local if mesh_node.parent is None else mesh_node.matrix_world
        mesh_node.parent = None

        # Perform removal of other nodes
        for obj_node in all_object_nodes:
            if obj_node != mesh_node:
                bpy.data.objects.remove(obj_node, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_node.select_set(True)
    bpy.context.view_layer.objects.active = mesh_node

    return mesh_node


def transform_object_origin(obj, set_origin_to_bottom=True):
    """
    Transform object to align with the scene, make the bottom point or center point of the object to be the origin

    Args:
        obj: blender object
    """
    bbox_min, bbox_max = scene_bbox(obj)

    new_origin = np.zeros(3)
    new_origin[0] = (bbox_max[0] + bbox_min[0]) / 2.
    new_origin[1] = (bbox_max[1] + bbox_min[1]) / 2.
    if set_origin_to_bottom:
        new_origin[2] = bbox_min[2]
    else:
        new_origin[2] = (bbox_max[2] + bbox_min[2]) / 2.

    all_object_nodes = [obj] + obj.children_recursive

    ## move the asset origin to the new origin
    for obj_node in all_object_nodes:
        if obj_node.data:
            me = obj_node.data
            mw = obj_node.matrix_world
            matrix = obj_node.matrix_world
            o = Vector(new_origin)
            o = matrix.inverted() @ o
            me.transform(Matrix.Translation(-o))
            mw.translation = mw @ o

    ## move all transform to origin (no need to since Empty objects have all been removed)
    # for obj_node in all_object_nodes:
    #    obj_node.matrix_world.translation = [0, 0, 0]
    #    obj_node.rotation_quaternion = [1, 0, 0, 0]

    bpy.ops.object.select_all(action="DESELECT")


def rotate_obj(obj, R):
    """
    Apply rotation matrix to blender object

    Args:
        obj: blender object
        R: (3, 3) rotation matrix
    """
    R = Matrix(R)
    obj.rotation_mode = 'QUATERNION'
    
    # Combine the rotations by matrix multiplication
    current_rotation = obj.rotation_quaternion.to_matrix().to_3x3()
    new_rotation_matrix = R @ current_rotation
    
    # Convert back to a quaternion and apply to the object
    obj.rotation_quaternion = new_rotation_matrix.to_quaternion()


def get_object_center_to_bottom_offset(obj):
    """
    Get the offset from the center to the bottom of the object

    Args:
        obj: blender object

    Returns:
        offset: (3,) offset
    """
    bbox_min, bbox_max = scene_bbox(obj)
    bottom_pos = np.zeros(3)
    bottom_pos[0] = (bbox_max[0] + bbox_min[0]) / 2.
    bottom_pos[1] = (bbox_max[1] + bbox_min[1]) / 2.
    bottom_pos[2] = bbox_min[2]
    offset = np.array(obj.location) - bottom_pos
    return offset


def insert_object(obj_path, pos, rot, scale=1.0, from_3DGS=False):
    """
    Insert object into the scene

    Args:
        obj_path: path to the object
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale of the object

    Returns:
        inserted_obj: blender object
    """
    inserted_obj = load_object(obj_path)
    merge_meshes(inserted_obj)
    inserted_obj = remove_empty_nodes_v1(inserted_obj)
    # inserted_obj = remove_empty_nodes_v2(inserted_obj)
    if not from_3DGS and not inserted_obj.name.startswith('chatsim'):
        normalize_scene(inserted_obj)
    transform_object_origin(inserted_obj, set_origin_to_bottom=False)  # set origin to the center for simulation
    inserted_obj.scale *= scale
    rotate_obj(inserted_obj, rot)
    bpy.context.view_layer.update()
    ## object origin is at center and pos represents the contact point, so we need to adjust the object position
    ## NOTE: we might have problem when rotation on either x or y axis is not 0
    if True:
        offset = get_object_center_to_bottom_offset(inserted_obj)
        inserted_obj.location = pos + offset
    else:
        inserted_obj.location = pos   # TODO: 3DGS might also adopt bottom point as position
    inserted_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # apply transform to allow simulation
    if from_3DGS:
        bpy.ops.object.shade_smooth()  # smooth shading for 3DGS objects
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    return inserted_obj


#########################################################
# Blender lighting setup
#########################################################
def add_env_lighting(env_map_path: str, strength: float = 1.0):
    """
    Add environment lighting to the scene with controllable strength.

    Args:
        env_map_path (str): Path to the environment map.
        strength (float): Strength of the environment map.
    """
    # Ensure that we are using nodes for the world's material
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create an environment texture node and load the image
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(env_map_path)

    # Create a Background node and set its strength
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength

    # Create an Output node
    out = nodes.new('ShaderNodeOutputWorld')

    # Link nodes together
    links = world.node_tree.links
    links.new(env.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], out.inputs['Surface'])


#########################################################
# Shadow catcher setup
#########################################################
def add_meshes_shadow_catcher(mesh_path=None, is_uv_mesh=False):
    """
    Add entire scene meshes as shadow catcher to the scene

    Args:
        mesh_path: path to the mesh file
        is_uv_mesh: whether the mesh is a UV textured mesh
    """
    # add meshes extracted from NeRF/3DGS as shadow catcher
    if mesh_path is None or not os.path.exists(mesh_path):
        AssertionError('meshes file does not exist')
    mesh = load_object(mesh_path)
    # mesh.is_shadow_catcher = True   # set True for transparent shadow catcher
    mesh.visible_diffuse = False      # prevent meshes light up the scene

    if not is_uv_mesh:
        raise NotImplementedError('Only UV textured meshes are supported for shadow catcher')

    bpy.ops.object.select_all(action="DESELECT")
    return mesh


#########################################################
# Others (Utility functions)
#########################################################
def set_hide_render_recursive(obj, hide_render):
    if obj.name.endswith('_proxy'):   # prevent proxy objects being rendered
        return
    obj.hide_render = hide_render
    for child in obj.children:
        set_hide_render_recursive(child, hide_render)


def set_visible_camera_recursive(obj, visible_camera):
    if obj.name.endswith('_proxy'):   # prevent proxy objects being rendered
        return
    obj.visible_camera = visible_camera
    for child in obj.children:
        set_visible_camera_recursive(child, visible_camera)


def delete_object_recursive(obj):
    for child in obj.children:
        delete_object_recursive(child)
    bpy.data.objects.remove(obj, do_unlink=True)


#########################################################
# Main function
#########################################################

def main_render(obj_files, scale, light_intensity, render_option, x, y, z, angle):
    # config_path = '/home/haoyuyh3/Documents/maxhsu/CS445_Final_Project_app/data/blender_cfg.json'
    config_path = './data/blender_cfg.json'
    config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    h, w = config['im_height'], config['im_width']
    K = np.array(config['K'])
    c2w = np.array(config['c2w'])
    scene_mesh_path = './data/mesh/bugatti.obj'
    scene_mesh_path = os.path.abspath(scene_mesh_path)
    if not os.path.exists(scene_mesh_path):
        print(f"File not found: {scene_mesh_path}")
    else:
        print(f"Loading file succ: {scene_mesh_path}")

    global_env_map_path ='./data/hdr/transforms_001/00000_rotate.exr'
    global_env_map_path = os.path.abspath(global_env_map_path)
    output_dir = './output'
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir) 

    # anti-aliasing rendering
    # upscale = 2.0
    # w = int(w * upscale)
    # h = int(h * upscale)
    # if len(K.shape) == 2:
    #     K[0, 0] *= upscale
    #     K[1, 1] *= upscale
    #     K[0, 2] *= upscale
    #     K[1, 2] *= upscale
    # else:
    #     for i in range(len(K)):
    #         K[i][0, 0] *= upscale
    #         K[i][1, 1] *= upscale
    #         K[i][0, 2] *= upscale
    #         K[i][1, 2] *= upscale

    setup_blender_env(w, h)
    scene_mesh = add_meshes_shadow_catcher(scene_mesh_path, is_uv_mesh=True)
    add_env_lighting(global_env_map_path, strength=light_intensity)     # TODO: this strength value adjustable for users

    cam = Camera(h, w, output_dir)
    cam.initialize_depth_extractor()  # initialize once
    cam_list = create_camera_list(c2w, K)

    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_end = len(c2w)  # TODO: unblock this to render the entire video


    ##### TODO: unit function for adding objects in the scene #####
    # obj_path = '/home/haoyuyh3/Documents/maxhsu/CS445_Final_Project_app/data/assets/basketball.glb'
    # obj_path = './data/assets/basketball.glb'
    # obj_path = os.path.abspath(obj_path)
    # obj_pos = np.array([0.20, -0.03, -0.45])
    # obj_rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # obj_scale = 0.09
    
    # object_mesh = insert_object(obj_path, obj_pos, obj_rot, obj_scale, from_3DGS=False)


    # for obj_file in obj_files:
    obj_path = obj_files

    obj_pos = np.array([float(x), float(y), float(z)])
    angle_rad = np.deg2rad(float(angle))
    obj_rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    object_mesh = insert_object(obj_path, obj_pos, obj_rot, scale, from_3DGS=False)

    ##### Render the scene #####
    RENDER_TYPE = 'SINGLE_VIEW' if render_option == "Single Image" else 'MULTI_VIEW'
    ##### Render the scene #####
    RENDER_TYPE = 'MULTI_VIEW'  # 'SINGLE_VIEW' or 'MULTI_VIEW'
    for FRAME_INDEX in range(scene.frame_start, scene.frame_end + 1):

        scene.frame_set(FRAME_INDEX)
        scene.cycles.samples = 32           # TODO: increase for higher quality but slower rendering
        bpy.context.view_layer.update()     # Ensure the scene is fully updated

        # Step 1: render only inserted objects
        set_visible_camera_recursive(object_mesh, True)
        scene_mesh.visible_camera = False
        if RENDER_TYPE == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_obj', dir_name_depth='depth_obj')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_obj', dir_name_depth='depth_obj')

        # Step 2: render only shadow catcher
        set_hide_render_recursive(object_mesh, True)
        scene_mesh.visible_camera = True
        if RENDER_TYPE == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_shadow', dir_name_depth='depth_shadow')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_shadow', dir_name_depth='depth_shadow')

        # Step 3: render all effects
        set_hide_render_recursive(object_mesh, False)
        set_visible_camera_recursive(object_mesh, True)
        scene_mesh.visible_camera = True
        if RENDER_TYPE == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_all', dir_name_depth='depth_all')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_all', dir_name_depth='depth_all')
            

if __name__ == "__main__":
    
    args = sys.argv[sys.argv.index("--") + 1:]

    if len(args) != 8:
        raise ValueError("Expected 8 arguments: obj_files, scale, light_intensity, render_option, x, y, z, angle")

    obj_files = args[0]
    scale = float(args[1])
    light_intensity = float(args[2])
    render_option = args[3]
    x = float(args[4])
    y = float(args[5])
    z = float(args[6])
    angle = int(args[7])
    
    main_render(
        obj_files=obj_files,
        scale=scale,
        light_intensity=light_intensity,
        render_option=render_option,
        x=x,
        y=y,
        z=z,
        angle=angle
    )