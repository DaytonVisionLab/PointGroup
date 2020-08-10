"""
Creates .ply data using .txt data. Splits .txt data into batches.

The directories containing many .txt files will be used to create 
directories containing two .ply files. The input directory is expected
to be formatted like the DALES aerial LiDAR dataset. The output files 
created by this module are compatible with PointGroup.
"""

import json
import argparse
import numpy as np
from pathlib import Path

from plyfile import PlyData, PlyElement

labels = {
    "ground": 1,
    "vegetation": 2, 
    "cars": 3, 
    "trucks": 4, 
    "powerlines": 5, 
    "fences": 6, 
    "poles": 7, 
    "buildings": 8 
}

def main(in_path: str, out_path: str):
    in_dir = Path(in_path)
    out_dir = Path(out_path)

    scenes = [scene for scene in in_dir.iterdir() if scene.is_dir()]

    for scene in scenes:
        annotations = scene / 'Annotations'
        try:
            instances = [instance for instance in annotations.iterdir() if instance.is_file()] 
        except FileNotFoundError:
            print("Invalid directory structure, see documentation.")
            continue
        out_scene = out_dir / scene.name
        
        if out_scene.is_dir():
            print("Skipping scene, already exists: {}".format(scene.name))
            continue

        print("Converting scene: {}".format(scene.name))
        out_scene.mkdir()     
    
        ply = []
        ply_labels = []
        point_instances = []
        
        for instance_number, instance in enumerate(instances):
            with instance.open(mode='r') as i:
                alpha = 0
                try:
                    label = labels[instance.name.split("_")[0]]
                except KeyError:
                    print("Skipping invalid file name: {}".format(instance.name))
                    continue
                
                lines = i.readlines()

                for line in lines:
                    point = [float(val) if "." in val else int(val) for val in line.split(" ")]
                    point.append(alpha)  
                    ply.append(tuple(point))
                    point.append(label)
                    ply_labels.append(tuple(point))
                    point_instances.append(instance_number)

        ply_array = np.array(ply, dtype=[
            ('x', 'f4'), 
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'), 
            ('green', 'u1'),
            ('blue', 'u1'),
            ('alpha', 'u1')])
        ply_labels_array = np.array(ply_labels, dtype=[
            ('x', 'f4'), 
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'), 
            ('green', 'u1'),
            ('blue', 'u1'),
            ('alpha', 'u1'),
            ('label', 'u2')])

        #Gridding
        grid_size = 9 #8x8 grid
        min_vertices = 155

        x_coords = np.array([coords[0] for coords in ply])
        y_coords = np.array([coords[1] for coords in ply])

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)         

        x1 = np.linspace(min_x, max_x, grid_size, endpoint=True)
        y1 = np.linspace(min_y, max_y, grid_size, endpoint=True)


        for xgrid in range(x1.shape[0]-1):
            for ygrid in range(y1.shape[0]-1):
                xcond = (x_coords >= x1[xgrid]) & (x_coords <= x1[xgrid+1])
                ycond = (y_coords >= y1[ygrid]) & (y_coords <= y1[ygrid+1])
                grid_ind = xcond & ycond
                grid_name = "_{}_{}".format(str(xgrid).zfill(2), str(ygrid).zfill(2))
                
                num_vertices = np.shape(ply_array[grid_ind])[0]

                if num_vertices > min_vertices:
                    ply_el = PlyElement.describe(ply_array[grid_ind], 'vertex')
                    ply_labels_el = PlyElement.describe( ply_labels_array[grid_ind], 'vertex')

                    out_ply = out_scene / ("scene" + scene.name + grid_name + "_vh_clean_2.ply")
                    out_label = out_scene / ("scene" + scene.name + grid_name + '_vh_clean_2.labels.ply')        

                    PlyData([ply_el]).write(str(out_ply))
                    PlyData([ply_labels_el]).write(str(out_label))

                    out_instances = out_scene / ("scene" + scene.name + grid_name + ".instances.json")

                    with open(out_instances, 'w') as i:
                        instances_dict = {"instances": [i for (i, v) in zip(point_instances, grid_ind) if v]}
                        json.dump(instances_dict, i) 

                    print("{} vertices saved to {}!".format(num_vertices, "scene" + scene.name + grid_name))

                else:
                    print("Skipping batch with {} vertices: {}".format(num_vertices, "scene" + scene.name + grid_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert aerial LiDAR to PointGroup compatible format.")
    parser.add_argument("-i", "--in_path", type=str, default="/home/vlab/Desktop/INSSEG/areas",
        help="Path to input directory containing aerial LiDAR scenes being converted.")
    parser.add_argument("-o", "--out_path", type=str, default="./ply_grid_8",
        help="Path to output directory for converted aerial LiDAR data.")
    args = parser.parse_args()
    
    main(args.in_path, args.out_path)
