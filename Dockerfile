#########################################################################
# Copyright 2025 Aion Sigma Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#########################################################################

FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV PROJECT_HOME=/opt/hermes
ENV HERMES_ENV=hermes-env

WORKDIR /opt/hermes/

RUN apt update && apt install -y python3 python3-pip python3-venv pandoc

RUN mkdir -p $PROJECT_HOME && cd $PROJECT_HOME && python3 -m venv $HERMES_ENV

COPY requirements.txt $PROJECT_HOME/

RUN chmod u+x $PROJECT_HOME/$HERMES_ENV/bin/activate

RUN /bin/bash -c "\
. $PROJECT_HOME/$HERMES_ENV/bin/activate \
&& pip install -r $PROJECT_HOME/requirements.txt \
&& playwright install \
&& playwright install-deps"

RUN mkdir -p $PROJECT_HOME/scripts

RUN cat /root/.profile

RUN echo "\n\nif (tty -s); then \n\
  source $PROJECT_HOME/$HERMES_ENV/bin/activate \n\
fi" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]