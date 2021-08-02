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
#from stardist import random_label_cmap
#from stardist.models import StarDist2D
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection
# from cytomine.models.property import Tag, TagCollection, PropertyCollection
from cytomine.utilities.software import parse_domain_list, str2bool, setup_classify, stringify


from PIL import Image
from keras.preprocessing import image
from keras.applications.densenet import DenseNet201

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import math

import argparse
import json
import logging



__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "0.1.0"
# Date created: 03 June 2021


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
        model_directory = os.path.join(base_path,'models/ModelDenseNet201')
        model_name = 'densenet201weights.best.hdf5'
        # model_name = 'weights.best.hdf5'
        print(model_directory +'/'+ model_name)
        print('Loading model.....')
        model = tf.keras.models.load_model(model_directory +'/'+ model_name)
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
                #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)

                # im=Image.open(roi_png_filename)
                img = tf.keras.preprocessing.image.load_img(roi_png_filename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                arr = tf.keras.preprocessing.image.img_to_array(img)
                arr = np.expand_dims(arr, axis=0)
                arr /= 255
                predictions.append(model.predict(arr))

                pred_labels = np.argmax(predictions, axis=-1)

                # roi_class_path=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png')



                if pred_labels[i][0]==0:
                    print("Class 0: Negative")
                    id_terms=conn.parameters.cytomine_id_c0_term
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==1:
                    print("Class 1: Weak")
                    id_terms=conn.parameters.cytomine_id_c1_term
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==2:
                    print("Class 2: Moderate")
                    id_terms=conn.parameters.cytomine_id_c2_term
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==3:
                    print("Class 3: Strong")
                    id_terms=conn.parameters.cytomine_id_c3_term
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

            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            print("Prediction time: ",end_time-start_prediction_time)

        conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])

    #with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        #run(cyto_job, cyto_job.parameters)
