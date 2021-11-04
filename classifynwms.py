# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


##==== inside a folder containing Dockerfile, run: sudo docker build -t cytomine/s_python_classifypncell ====##

import sys
import numpy as np
import os
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
#from csbdeep.utils import Path, normalize
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection
# from cytomine.models.property import Tag, TagCollection, PropertyCollection
# from cytomine.utilities.software import parse_domain_list, str2bool, setup_classify, stringify


from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet201
pretrained_model = DenseNet201(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
pretrained_model.trainable = False

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import math

import argparse
import json
import logging
# import pathlib



__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.0"
# Date created: 03 June 2021 (modified on 15 Oct 2021 for prune)

# def densemodel():
#     data_augmentation = tf.keras.Sequential([
#         tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#         tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
#         tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
#     ])
#     tensor = tf.keras.Input((224, 224, 3))
#     x = tf.cast(tensor, tf.float32)
#     x = tf.keras.applications.densenet.preprocess_input(
#         x, data_format=None)
#     x = data_augmentation(x)
#     x = pretrained_model(x, training=False)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(256)(x)
#     x = tf.nn.relu(x)
#     x = tf.keras.layers.Dropout(0.4)(x)
#     x = tf.keras.layers.Dense(4)(x)
#     x = tf.nn.softmax(x)
#     model = tf.keras.Model(tensor, x)
#     model.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss=tf.keras.losses.CategoricalCrossentropy(),
#                   metrics=['accuracy'])
#     return model


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
    # with Cytomine(argv) as conn:
        print(conn.parameters)

        conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")
        base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
        working_path = os.path.join(base_path,str(conn.job.id))


        terms = TermCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)
        conn.job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
        print(terms)

        start_time=time.time()

#         model_directory = os.path.join(base_path,'models/ModelDenseNet201')
#         model_directory = working_path
        model_directory = '/models/ModelDenseNet201'

        # model_name = 'densenet201weights.best.h5'
#         model_dir = pathlib.Path("weights_float16/")
#         
#        print('current working dir:',pathlib.Path.cwd())
#         model_dir = pathlib.Path.cwd()
        model_name = 'weights.best.h5'
#         model_file = pathlib.Path("model_quant_f16.tflite")
#         model_file = model_dir/"model_quant_f16.tflite"  
#         model_file = model_dir/model_name
#         model_name = 'model_quant_f16.tflite'

        
        print(model_directory +'/'+ model_name)
        print('Loading model.....')
#         print(model_file)
#         model = densemodel()
#         model.load_weights(model_file)
#         model.load_weights(model_directory +'/'+ model_name)
#         model.load_weights(working_path +'/'+ model_name)

        
        model = tf.keras.models.load_model(model_directory +'/'+ model_name, compile = False)
#         model = tf.keras.models.load_model(model_name)
#         model = tf.saved_model.load(model_name)

#         model_interpreter = tf.lite.Interpreter(model_path=model_directory +'/'+ model_name)
#         model_interpreter = tf.lite.Interpreter(model_path=str(model_file))
#         model_interpreter.allocate_tensors()
        

        print('Model successfully loaded!')
        IMAGE_CLASSES = ['c0', 'c1', 'c2', 'c3']
        IMAGE_WIDTH, IMAGE_HEIGHT = (224, 224)
       
        # #Loading pre-trained Stardist model
        # np.random.seed(17)
        # lbl_cmap = random_label_cmap()
        # #Stardist H&E model downloaded from https://github.com/mpicbg-csbd/stardist/issues/46
        # #Stardist H&E model downloaded from https://drive.switch.ch/index.php/s/LTYaIud7w6lCyuI
        # model = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')   #use local model file in ~/models/2D_versatile_HE/

        #Select images to process
        images = ImageInstanceCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)
        conn.job.update(status=Job.RUNNING, progress=2, statusComment="Images gathered...")
        
        list_imgs = []
        if conn.parameters.cytomine_id_images == 'all':
            for image in images:
                list_imgs.append(int(image.id))
        else:
            list_imgs = [int(id_img) for id_img in conn.parameters.cytomine_id_images.split(',')]
            print(list_imgs)



        #Go over images
        conn.job.update(status=Job.RUNNING, progress=10, statusComment="Running PN classification on image...")
        #for id_image in conn.monitor(list_imgs, prefix="Running PN classification on image", period=0.1):
        for id_image in list_imgs:
            print('Current image:', id_image)
            roi_annotations = AnnotationCollection()
            roi_annotations.project = conn.parameters.cytomine_id_project
            roi_annotations.term = conn.parameters.cytomine_id_cell_term
            roi_annotations.image = id_image #conn.parameters.cytomine_id_image
            roi_annotations.job = conn.parameters.cytomine_id_annotation_job
            roi_annotations.user = conn.parameters.cytomine_id_user_job
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            print(roi_annotations)

            # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
            # print(roi_path)

            # number_class=4
            # for i in range(number_class):
            #     try:
            #         os.mkdir(os.path.join(roi_path,str(i)))
            #     except:
            #         pass


            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            # for roi in roi_annotations:
            for i, roi in enumerate(roi_annotations):
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
                print("----------------------------Cells------------------------------")
                roi_geometry = wkt.loads(roi.location)
                # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)
                minx=roi_geometry.bounds[0]
                miny=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                print(roi_path)
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                conn.job.update(status=Job.RUNNING, progress=20, statusComment=roi_png_filename)
                print("roi_png_filename: %s" %roi_png_filename)
                roi.dump(dest_pattern=roi_png_filename,alpha=True)
#                 roi.dump(dest_pattern=roi_png_filename, mask=True, alpha=True)
                #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)

                # im=Image.open(roi_png_filename)

                # img = tf.keras.preprocessing.image.load_img(roi_png_filename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                # im_arr = tf.keras.preprocessing.image.img_to_array(img)
#                 im = cv2.imread(roi_png_filename).astype('float32')
                im = cv2.imread(roi_png_filename)
                im_arr = np.array(im)
                im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
                im_arr = cv2.resize(im_arr, (224, 224))
                im_arr = np.expand_dims(im_arr, axis=0)
                # im_arr /= 255


#                 input_index = model_interpreter.get_input_details()[0]["index"]
#                 output_index = model_interpreter.get_output_details()[0]["index"]

#                 model_interpreter.set_tensor(input_index, im_arr)
#                 model_interpreter.invoke()
#                 predictions = model_interpreter.get_tensor(output_index)
                predictions.append(model.predict(im_arr))
                pred_labels = np.argmax(predictions, axis=-1)
        
                print("Prediction:", predictions)

                pred_labels = np.argmax(predictions, axis=-1)
                print("PredLabels:", pred_labels)            
                img_all.append(roi_png_filename)
                # print(img_all)
                
                
                pred_all.append(pred_labels)
                print(pred_all)

                # roi_class_path=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png')

                if pred_labels[i][0]==0:
                    print("Class 0: Negative")
                    id_terms=conn.parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==1:
                    print("Class 1: Weak")
                    id_terms=conn.parameters.cytomine_id_c1_term
                    pred_c1=pred_c1+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==2:
                    print("Class 2: Moderate")
                    id_terms=conn.parameters.cytomine_id_c2_term
                    pred_c2=pred_c2+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==3:
                    print("Class 3: Strong")
                    id_terms=conn.parameters.cytomine_id_c3_term
                    pred_c3=pred_c3+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class3/'+str(roi.id)+'.png'),alpha=True)


                cytomine_annotations = AnnotationCollection()

                annotation=roi_geometry

                # tags.append(TagDomainAssociation(Annotation().fetch(id_image, tag.id))).save()

                # association = append(TagDomainAssociation(Annotation().fetch(id_image, tag.id))).save()
                # print(association)


                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=conn.parameters.cytomine_id_project,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)

                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()

            # print("prediction all:", pred_all)
            # print(pred_labels)

            # print("prediction c0:", pred_c0)
            # print("prediction c1:", pred_c1)
            # print("prediction c2:", pred_c2)
            # print("prediction c3:", pred_c3)
            pred_all=[pred_c0, pred_c1, pred_c2, pred_c3]
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1+pred_c2+pred_c3
            print("pred_total:",pred_total)
            pred_positive=pred_c1+pred_c2+pred_c3
            print("pred_positive:",pred_positive)
            pred_positive_100=pred_positive/pred_total*100
            print("pred_positive_100:",pred_positive_100)

            if pred_positive_100 == 0:
                proportion_score = 0
            elif pred_positive_100 < 1:
                proportion_score = 1
            elif pred_positive_100 >= 1 and pred_positive_100 <= 10:
                proportion_score = 2
            elif pred_positive_100 >= 11 and pred_positive_100 <= 33:
                proportion_score = 3
            elif pred_positive_100 >= 34 and pred_positive_100 <= 66:
                proportion_score = 4
            elif pred_positive_100 >= 67:
                proportion_score = 5

            if pred_positive_100 == 0:
                intensity_score = 0
            elif im_pred == 1:
                intensity_score = 1
            elif im_pred == 2:
                intensity_score = 2
            elif im_pred == 3:
                intensity_score = 3

            allred_score = proportion_score + intensity_score
            print('Proportion Score: ',proportion_score)
            print('Intensity Score: ',intensity_score)            
            print('Allred Score: ',allred_score)
            
            
        end_time=time.time()
        print("Execution time: ",end_time-start_time)
        print("Prediction time: ",end_time-start_prediction_time)

    conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])

    #with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        #run(cyto_job, cyto_job.parameters)
