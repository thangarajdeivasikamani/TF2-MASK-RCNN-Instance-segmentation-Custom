# TF2-MASK-RCNN-Instance-segmentation-Custom Dataset (Fashion)
- This repo contains Tensorflow 2 based MASK RCNN colab notebook to train the custom data and flask api to detect the images using browser.
 #### What is instance segmentation and how is different from semantic segmentation?
- Semantic Segmentation detects all the objects present in an image at the pixel level. Outputs regions with different classes or objects

- Semantic segmentation groups pixels in a semantically meaningful way. Pixels belonging to a sunclass, bag, shirts, pants, skirts, shoe or hat are grouped separately.
- instance Segmentation is identifying each object instance for every known object within an image.
 #### What’s different in Mask R-CNN and Faster R-CNN?
- Mask R-CNN has an additional branch for predicting segmentation masks on each Region of Interest (RoI) in a pixel-to pixel manner. it hass three output bounding box,class and mask info.
- Faster R-CNN is not designed for pixel-to-pixel alignment. It has two outputs bounding box and class
- ![image](https://user-images.githubusercontent.com/46878296/171320575-249602f3-a15a-4500-a262-d4f1863f46d5.png)
- source:https://catchzeng.medium.com/train-a-custom-image-segmentation-model-using-tensorflow-object-detection-api-mask-r-cnn-a8cbfd2321e0


#### Difference B/w RCNN,FASTRCNN and FASTER RCNN:

  -Refer https://stackoverflow.com/questions/43402760/object-detection-with-r-cnn

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
    
 - 5.Prepare the data
     - Download the Tool from  https://github.com/wkentaro/labelme#installation and install
     -
 - 6.Annonate the data (using polycon) and save the json file format
       ![image](https://user-images.githubusercontent.com/46878296/171323017-010257d7-d281-4fd0-b023-2f4398613f7b.png)
    
 - 7.Sample Json file
     ![image](https://user-images.githubusercontent.com/46878296/171332504-50cc541b-ec69-43cd-98e3-579a94fd2e4b.png)


 - 8.Go to root folder e.g (Fashion_instance) and create the below folder strcutures

      - !mkdir workspace
      - !mkdir workspace/training_demo/
      - !mkdir workspace/training_demo/annotations/
      - !mkdir workspace/training_demo/images/
      - !mkdir workspace/training_demo/images/train/
      - !mkdir workspace/training_demo/images/test/
      - !mkdir workspace/training_demo/pre-trained-models/ -
      - !mkdir workspace/training_demo/export_models/
      - !mkdir workspace/training_demo/models/

 - 9.Copy the Train and Test image and alogn with annonation files into below folder
 
      - Fashion_instance/workspace/training_demo/images
      - Also copy the labelmap.pbtxt to annonation folder
          
 - 10.install labelme software !pip install labelme
 
 - 11.Create Train and Test json for that from models/objectdetection folder run the below commands

      - !python labelme2coco.py  train --output Fashion_instance/workspace/training_demo/annotations/train.json
      - !python labelme2coco.py  test --output Fashion_instance/workspace/training_demo/annotations/test.json
      - Ensure that both json file created.
      
  - 12.Install pycocotools
  
      - !pip uninstall pycocotools
      - !pip install pycocotools --no-binary pycocotools

  - 13.Create the record
  
      -  !python create_coco_tf_record.py --logtostderr \
      - --train_image_dir=Fashion_instance/workspace/training_demo/images/train \
      - --test_image_dir=Fashion_instance/workspace/training_demo/images/test \
      - --train_annotations_file=Fashion_instance/workspace/training_demo/annotations/train.json \
      - --test_annotations_file=Fashion_instance/workspace/training_demo/annotations/test.json \
      - --include_masks=True \
      - --output_dir=Fashion_instance/workspace/training_demo/annotations/
      - Ensure that Tets and Train TF record created under Fashion_instance/workspace/training_demo/annotations/ folder 

   - 14.Go to Pretained model folder and download the mask RCNN pretained model info from Tensofrflow website and extarct
   
          - !wget http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
          - !tar -xvf mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

   - 15.Downlaod the config file into annotation folder 
   
          - !wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu- 8.config

   - 16.Update the config file as below 
   
         - batch_size = 1
         - num_steps = 100
         - num_eval_steps = 10000
         - num_classes=10
         - # Manully edit 
         - total_steps: 5000
         - warmup_steps: 5000
         - # Evalue steps
         - batch_size: 1
         - height: 640
         - width: 640
         - first_stage_max_proposals: 100
         - train_record_path = 'Fashion_instance/workspace/training_demo/annotations/train.record'
         - test_record_path = 'Fashion_instance/workspace/training_demo/annotations/test.record'
         - labelmap_path = 'Fashion_instance/workspace/training_demo/annotations/labelmap.pbtxt'
         - fine_tune_checkpoint = 'Fashion_instance/workspace/training_demo/pre-trained-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/checkpoint/ckpt-0'

  - 17.Uninstall the existing lib and install
      
      - !pip uninstall opencv-python-headless==4.5.5.62
      - !pip install opencv-python-headless==4.5.2.52
      - !pip uninstall pycocotools
      - !pip install pycocotools --no-binary pycocotools
      
   ## Training the Model
   
      - !python Fashion_instance/models/research/object_detection/model_main_tf2.py \
      - --model_dir='Fashion_instance/workspace/training_demo/models' \
      - --pipeline_config_path='Fashion_instance/workspace/training_demo/annotations/model_config.config'
   
  ### Export the Trainined model
  
      - !python exporter_main_v2.py --input_type image_tensor \
      - --pipeline_config_path Fashion_instance/workspace/training_demo/annotations/model_config.config \
      - --trained_checkpoint_dir Fashion_instance/workspace/training_demo/models \
      - --output_directory Fashion_instance/workspace/training_demo/export_models
      
 ### Tensorflow board view
     -  %load_ext tensorboard
     -  %tensorboard --logdir 'Fashion_instance/workspace/training_demo/models/train'
 
 ### Prediction ouptut:
 
 ![image](https://user-images.githubusercontent.com/46878296/171359746-bc4f8d05-83b1-40a1-8988-3449e9a96c5a.png)
 ![image](https://user-images.githubusercontent.com/46878296/171364472-b2bab3f2-5a5c-4194-8ae8-7c6d8b526445.png)

### Webapi output:
![image](https://user-images.githubusercontent.com/46878296/171366821-fa5fff6a-742f-42af-8416-fc9efd137a2a.png)


