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

RUN mkdir -p /models && \
    cd /models && \
    mkdir -p ModelDenseNet201

# add model
ADD densenet201weights.best.h5 /models/ModelDenseNet201/densenet201weights.best.h5
RUN chmod 444 /models/ModelDenseNet201/densenet201weights.best.h5


# Install scripts
ADD descriptor.json /app/descriptor.json
RUN mkdir -p /app
ADD classifynwms.py /app/classifynwms.py

ENTRYPOINT ["python", "/app/classifynwms.py"]
