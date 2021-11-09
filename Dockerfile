# * Copyright (c) 2009-2020. Authors: see NOTICE file.
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

FROM cytomine/software-python3-base

#INSTALL
RUN apt-get update
RUN pip install tensorflow==2.5.0
RUN pip install keras
RUN pip install matplotlib
RUN pip install numpy
RUN pip install shapely
RUN pip install tifffile
RUN pip install h5py
# RUN pip install pathlib

RUN mkdir -p /models && \
    cd /models && \
    mkdir -p ModelDenseNet201 && \
    cd /ModelDenseNet201 && \
    mkdir -p /variables 

ADD keras_metadata.pb /models/ModelDenseNet201/keras_metadata.pb
ADD saved_model.pb /models/ModelDenseNet201/saved_model.pb

ADD variables.data-00000-of-00001 /models/ModelDenseNet201/variables/variables.data-00000-of-00001
ADD variables.index /models/ModelDenseNet201/variables/variables.index



# ADD weights.best.hdf5 /models/ModelDenseNet201/weights.best.hdf5
# RUN chmod 444 /models/ModelDenseNet201/weights.best.hdf5

# RUN mkdir -p /weights_float16     
# ADD /weights_float16/model_quant_f16.tflite /weights_float16/model_quant_f16.tflite
# RUN chmod 444 /weights_float16/model_quant_f16.tflite

# Install scripts
ADD descriptor.json /app/descriptor.json
RUN mkdir -p /app
ADD classifynwms.py /app/classifynwms.py
# ADD weights.best.h5 /app/weights.best.h5

ENTRYPOINT ["python", "/app/classifynwms.py"]

# RUN mkdir /app
# COPY . .
# WORKDIR /app
# ENTRYPOINT ["python","/app/classifynwms.py"]

