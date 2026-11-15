import bpy
import bmesh
import math
import numpy as np

def generate_complete_seahorse_scene():
    mesh_name = "SeahorseTailTerrain"
    mat_name = "SeahorseGoldTextured"
    img_name = "SeahorseFractalTexture"

    # 1. Clean up old objects, meshes, images, and materials to start fresh
    if mesh_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[mesh_name], do_unlink=True)
    if mesh_name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[mesh_name], do_unlink=True)
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])

    # 2. Procedurally Generate the Seahorse Valley Texture Map (1024x1024 RGBA)
    width, height = 1024, 1024
    xmin, xmax = -0.75, -0.73
    ymin, ymax = 0.13, 0.15
    max_iter = 128

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = x[:, None] + 1j * y[None, :]
    z = np.zeros_like(c)
    counts = np.zeros(c.shape, dtype=float)
    
    mask = np.ones(c.shape, dtype=bool)
    for i in range(max_iter):
        if not mask.any():
            break
        z[mask] = z[mask]**2 + c[mask]
        diverged = (abs(z) > 2.0) & mask
        counts[diverged] = i + 1 - np.log2(np.maximum(1.0, np.log(abs(z[diverged]))))
        mask[diverged] = False
        
    counts[mask] = max_iter
    norm = counts / max_iter
    norm = np.power(norm, 0.6)
    
    # Sunset / Gold & Crimson color gradient
    r = np.clip(1.2 * np.sin(norm * math.pi * 0.8) + 0.3 * norm, 0, 1)
    g = np.clip(0.8 * norm + 0.2 * np.cos(norm * math.pi), 0, 1)
    b = np.clip(0.4 * norm**2, 0, 1)
    a = np.ones_like(r)
    
    img_arr = np.stack([r, g, b, a], axis=-1)
    img_arr = (img_arr * 255).astype(np.uint8)
    
    img = bpy.data.images.new(img_name, width=width, height=height)
    img.pixels = img_arr[:, :, :].ravel() / 255.0
    img.pack()

    # 3. Generate the 3D Terrain Mesh via bmesh
    resolution = 640
    span = 6.0
    height_scale = 3.0
    base_thickness = 0.25

    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    dx = (xmax - xmin) / (resolution - 1)
    dy = (ymax - ymin) / (resolution - 1)

    verts_grid = [[None for _ in range(resolution)] for _ in range(resolution)]

    for iy in range(resolution):
        py_y = ymin + iy * dy
        for ix in range(resolution):
            px_x = xmin + ix * dx
            pc = complex(px_x, py_y)
            pz = 0j
            piters = 0
            
            while abs(pz) <= 2.0 and piters < max_iter:
                pz = (pz * pz) + pc
                piters += 1

            if piters == max_iter:
                z_height = height_scale
            else:
                smooth_val = piters + 1 - math.log(math.log(max(abs(pz), 1.0001))) / math.log(2.0)
                norm_h = max(0.0, min(1.0, smooth_val / max_iter))
                z_height = base_thickness + (norm_h ** 1.4) * (height_scale - base_thickness)

            wx = (ix / (resolution - 1) - 0.5) * span
            wy = (iy / (resolution - 1) - 0.5) * span
            verts_grid[iy][ix] = bm.verts.new((wx, wy, z_height))

    bm.verts.ensure_lookup_table()

    for iy in range(resolution - 1):
        for ix in range(resolution - 1):
            v1 = verts_grid[iy][ix]
            v2 = verts_grid[iy][ix + 1]
            v3 = verts_grid[iy + 1][ix + 1]
            v4 = verts_grid[iy + 1][ix]
            bm.faces.new((v1, v2, v3, v4))

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    for poly in mesh.polygons:
        poly.use_smooth = True

    # 4. Assign Normal U and V Coordinates
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    min_x = min(v.co.x for v in mesh.vertices)
    max_x = max(v.co.x for v in mesh.vertices)
    min_y = min(v.co.y for v in mesh.vertices)
    max_y = max(v.co.y for v in mesh.vertices)
    span_x = max_x - min_x if max_x != min_x else 1.0
    span_y = max_y - min_y if max_y != min_y else 1.0

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert_idx = mesh.loops[loop_index].vertex_index
            co = mesh.vertices[vert_idx].co
            
            u = (co.x - min_x) / span_x
            v = (co.y - min_y) / span_y
            
            uv_layer[loop_index].uv = (u, v)

    # 5. Build and Apply PBR Gold Material with Mapping Node (Rotation Z: 90°, Scale Y: -1.0)
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Material Output Node
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (800, 0)

    # Principled BSDF Node
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (450, 0)
    node_bsdf.inputs['Metallic'].default_value = 0.94
    node_bsdf.inputs['Roughness'].default_value = 0.16

    # Image Texture Node
    node_tex = nodes.new(type='ShaderNodeTexImage')
    node_tex.location = (-150, 0)
    node_tex.image = img

    # Mapping Node - Preset Rotation Z to 90° and Scale Y to -1.0
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_mapping.location = (-450, 0)
    node_mapping.inputs['Rotation'].default_value[2] = math.radians(90.0)
    node_mapping.inputs['Scale'].default_value[1] = -1.0

    # Texture Coordinate Node
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_tex_coord.location = (-750, 0)

    # Bump Node
    node_bump = nodes.new(type='ShaderNodeBump')
    node_bump.location = (150, -200)
    node_bump.inputs['Strength'].default_value = 0.05

    # Connect Nodes
    links.new(node_tex_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex.inputs['Vector'])
    links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_tex.outputs['Color'], node_bump.inputs['Height'])
    links.new(node_bump.outputs['Normal'], node_bsdf.inputs['Normal'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # 6. Force Viewport Shading to Material Preview so it's visible immediately
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("Successfully generated Seahorse Valley terrain with Mapping node (Rotation Z: 90°, Scale Y: -1.0)!")

if __name__ == "__main__":
    generate_complete_seahorse_scene()