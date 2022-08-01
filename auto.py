import shutil
import os
import hdf5storage
mat = hdf5storage.loadmat("imagelabels.mat")

labels = mat["labels"][0]

for label in labels:
    path = os.listdir("102segmentations (1)\segmim")[0]
    print(path)
    shutil.move(rf"102segmentations (1)\segmim\{path}", r"dataset\train\{}".format(label))
    



