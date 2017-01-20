ICON is a software for training a neural network for 
for image segmentation tasks.  It supports MLP, CNN,
and UNET.  The UNET is not fully implemented in this
version of the code.

The UI runs on a web browser while the classifiers runs
on a server with GPU

![alt tag](https://github.com/Rhoana/icon/blob/master/screenshots/segmentation.png)

# REQUIRED PACKAGES
cython
h5py
hdf5
jpeg
keras
libpng
libtiff
mahotas
matplotlib
numpy
opencv
pandas
pil 
pillow
scikit-image 
scikit-learn
scipy
sqlite
theano
tornado

# EXECUTION

1. Run install.sh once, to setup the system 
   (This should be done on a linux system)

2. Start the web server by running:
   sh web.sh

3. Start the training thread by running:
   sh train.sh

4. Start the segmentation thread by running:
   sh segment.sh

5. Access the UI by launching the following URL
   on a browser:
   http://localhost:8888/browse

   Then select a project from the drop down list.
   Press the start button to activate a project
   or stop to deactivate.  Only one project can
   be active at a time. 
