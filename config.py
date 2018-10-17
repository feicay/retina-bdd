import numpy as np 

class cfg():
    width = 640
    height = 384
    batch = 64
    subvision = 8
    lr = 0.001
    max_epoch = 100
    dataset = 'bdd'
    trainlist = '/home/adas/data/bdd100k/train.txt'
    vallist = '/home/adas/data/bdd100k/val.txt'
    testlist = '/home/adas/data/bdd100k/test.txt'
    classes = 10
    box_len = 4
    num_anchor = 9
    anchors = np.array([[0.0134, 0.0238],
                        [0.0190, 0.0380],
                        [0.0280, 0.0660],
                        [0.0410, 0.0760],
                        [0.0500, 0.1080],
                        [0.0890, 0.1680],
                        [0.1400, 0.1910],
                        [0.2210, 0.3180],
                        [0.3400, 0.5220]])