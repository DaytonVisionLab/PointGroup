'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse

# Map relevant classes to {0,1,...,7}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
point_files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
if opt.data_split != 'test':
    point_label_files = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
    instance_files = sorted(glob.glob(split + '/*.instances.json'))
    assert len(point_files) == len(point_label_files)
    assert len(point_label_files) == len(instance_files)

def create_test_data(ply):
    '''
    Creates inst_nostuff.pth file for each test .ply file.

    ply: Name of the .ply file being used. e.g. "scene0000_0000_00_00_vh_clean_2.ply"
    '''
    print(ply)

    data = plyfile.PlyData().read(ply)
    points = np.array([list(x) for x in data.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), ply[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + ply[:-15] + '_inst_nostuff.pth')


def create_train_data(ply):
    '''
    Creates inst_nostuff.pth file for each train/val .ply file.

    ply: Name of the .ply file being used. e.g. "scene0000_0000_00_00_vh_clean_2.ply"
    '''
    labels_file = ply[:-3] + 'labels.ply'
    instances_file = ply[:-15] + '.instances.json'
    print(ply)

    data = plyfile.PlyData().read(ply)
    points = np.array([list(x) for x in data.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    data_labels = plyfile.PlyData().read(labels_file)
    sem_labels = remapper[np.array(data_labels.elements[0]['label'])]
#    sem_labels = np.array(data_labels.elements[0]['label'])

    with open(instances_file) as jsondata:
        data_instances = json.load(jsondata)
        instance_labels = np.array(data_instances['instances'])

    torch.save((coords, colors, sem_labels, instance_labels), ply[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + ply[:-15] + '_inst_nostuff.pth')

#for fn in files:
#    f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(create_test_data, point_files)
else:
    p.map(create_train_data, point_files)
p.close()
p.join()
