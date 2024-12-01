
# this is codebase 3. 
-> we want to prove the generative ability in our architecture 
-> if it predicts features, it should predict rgb also. 
    -> we tried earlier, but in column T, there was latent feature earlier. 
        -> that did not work. 
    -> now, we need to check the image in T, and rgb as output. hopefully it should work, o krishna!!!
-> also increased the learning rate, the network took a lot of time to converge last time,. 
# neural perception field

#codes 
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

+ to do:
    - add rgb head also. 
        - what to predict since the patch size becomes lower. averaging does not  appear to work that well.
    - instead of distilling dino, can i use mvitv2 lower layers. 
    - multi scale aliasing. (how to generate images of a single image at different scales)
        - take insights into how nerfs handle this. 

