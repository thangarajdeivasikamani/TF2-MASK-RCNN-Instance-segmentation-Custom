# TF2-MASK-RCNN-Instance-segmentation-Custom
- This repo contains Tensorflow 2 based MASK RCNN colab notebook to train the custom data and flask api to detect the images using browser.
- What is instance segmentation and how is different from semantic segmentation?
- Semantic Segmentation detects all the objects present in an image at the pixel level. Outputs regions with different classes or objects

- Semantic segmentation groups pixels in a semantically meaningful way. Pixels belonging to a sunclass, bag, shirts, pants, skirts, shoe or hat are grouped separately.
- instance Segmentation is identifying each object instance for every known object within an image.
 #### What’s different in Mask R-CNN and Faster R-CNN?
- Mask R-CNN has an additional branch for predicting segmentation masks on each Region of Interest (RoI) in a pixel-to pixel manner. it hass three output bounding box,class and mask info.
- Faster R-CNN is not designed for pixel-to-pixel alignment. It has two outputs bounding box and class
- ![image](https://user-images.githubusercontent.com/46878296/171320575-249602f3-a15a-4500-a262-d4f1863f46d5.png)
- source:https://catchzeng.medium.com/train-a-custom-image-segmentation-model-using-tensorflow-object-detection-api-mask-r-cnn-a8cbfd2321e0

#### Dataset
- https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection
#### Steps:

- 1.Login into google colab.
- 2.Install 
     - ! pip install --upgrade tf-models-official==2.4.0
     - ! pip install --upgrade tensorflow==2.4.1
     - ! pip install tensorflow-io==0.17.0
     -  Clone the TF2 models from https://github.com/tensorflow/models
     -  Go to /models/research/object_detection/packages/tf2/setup.py and remove the below lines in setup.py and save the file
     -  tf-models-official>=2.5.1
     -  tensorflow_io
- 3.Compile protobufs and install the object_detection package

     - %%bash
     - cd models/research/
     - protoc object_detection/protos/*.proto --python_out=
     - Go to /models/research/object_detection/packages/tf2 folder and run
     - pip install .

 - 4.Test your Envoriment Installation
    
     - Go to /models/research/object_detection
     - run !python builders/model_builder_tf2_test.py
 - 5. Prepare the data
     - Download the Tool from  https://github.com/wkentaro/labelme#installation and install
 - 6. annonate the data (using polycon) and save the json file format
    ![image](https://user-images.githubusercontent.com/46878296/171323017-010257d7-d281-4fd0-b023-2f4398613f7b.png)
    
 - 7. Sample Json file
     ![image](https://user-images.githubusercontent.com/46878296/171332504-50cc541b-ec69-43cd-98e3-579a94fd2e4b.png)


    
