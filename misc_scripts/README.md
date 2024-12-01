## training code for apm 

1_extract_coco_features.py for a resizing coco images to 448 by 448. dumps a 1024 (32^2) feature vector. 

2_read_numpy.py reads it. 

dataloader.py -> reads these numpy files, images, runs a max pool operation on it 

model.py ->
    takes an image. runs 2d conv on it. unrolls it into one dimension and gets the information into the column 
    the column is replicated across locations, and positional encodings are concatenated 
    the nerf is queried and trained to predict the perceptual features in the network 
    loss is a simple perceptual loss between predicted features and dumped dino features. 

train.py -> to train the network, the weights are dumped in the checkpoints directory. 

inference.py -> takes entire coco val set, and runs npf on it. visualizes the islands 

interpolate.py -> given two images, a conv on them shall yield two summary vectors (dna). performs linear interpolation on the images for certain 
                   number of iterations, and then forms the islands. 


the predicted features could thus be used for any downstream perception task. 
