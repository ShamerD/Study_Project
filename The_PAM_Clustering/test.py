import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
from PAM import PAM
from time import time

dir_path = os.path.dirname(os.path.realpath(__file__))
new_test_dir = os.path.join(dir_path, "imgs_clustered")

if not os.path.exists(new_test_dir):
    os.mkdir(new_test_dir)  # create dir for test results

for img_path in glob(os.path.join(dir_path, 'imgs', '*')):  # get imgs
    img = np.array(Image.open(img_path))[:, :, :3].astype('int64')
    file_name = os.path.basename(img_path)
    img_vectorised = img.reshape((-1, 3))  # get (n_pixels, 3)

    start = time()
    c, C, totalDist = PAM(img_vectorised, 3)  # 3 clusters, dist - manhattan
    pam_time = time() - start
    print("PAM executed in %.6f" % pam_time)

    img_new = np.zeros_like(img_vectorised)
    for i in range(img_new.shape[0]):
        img_new[i] = img_vectorised[C[i]]
    img_new = img_new.reshape(img.shape).astype('uint8')  # clustered image

    img_name = "clustered_" + os.path.split(img_path)[-1]
    im = Image.fromarray(img_new).convert('RGB')
    im.save(os.path.join(new_test_dir, img_name))

    fig = plt.figure(figsize=(16, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)  # original
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_new)  # clustered
    plt.show()
