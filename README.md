# End-to-end computer vision bootcamp

This repository contains the material for the **end-to-end computer vision** bootcamp, the goal of which is to build a complete end-to-end computer vision pipeline for an object detection application. This bootcamp will introduce participants to multiple NVIDIA SDKs, most notably TAO Toolkit, TensorRT, DeepStream, and Triton Inference Server. Participants will also have hands-on experience in data preprocessing, model training, optimization, and deployment at scale.

The content is structured in five modules, plus an introductory notebook:
- Welcome to **end-to-end computer vision** bootcamp
- Lab 1: Data labeling and preprocessing
- Lab 2: Object detection using TAO YOLOv4
- Lab 3: Model deployment with DeepStream
- Lab 4: Model deployment with Triton Inference Server
- Lab 5: Measure object size using OpenCV

## Tutorial duration

The total bootcamp material would take approximately 8.5 hours. It is recommended to divide the teaching of the material into two days, covering the first two notebooks in one session and the last three in the next section.

## Running using Singularity

To run the material using singularity containers, 

To build the TAO Toolkit singularity container, run: `singularity build --fakeroot --sandbox tao.simg Singularity_tao`

To build the Triton Inference Server singularity container for the client, run: `singularity build --fakeroot --sandbox triton_client.simg Singularity_triton`

To build the DeepStream singularity container, run: `singularity build --fakeroot --sandbox deepstream.simg Singularity_deepstream`

Then, run the first container with: `singularity run --fakeroot --nv -B ~/End_to_end_CV/workspace:/workspace/tao-experiments tao.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/tao-experiments`

The `-B` flag mounts local directories in the container filesystem and ensures changes are stored locally in the project folder. Open jupyter lab in browser: http://localhost:8888 

You may now start working on the lab by clicking on the `Start_here.ipynb` notebook.

When you are done with `1.Data_labeling_and_preprocessing.ipynb` and `2.Object_detection_using_TAO_YOLOv4.ipynb`, shut down jupyter lab by selecting `File > Shut Down` in the top left corner, then shut down the Singularity container by typing `exit` or pressing `ctrl d` in the terminal window.

You may now activate the Triton Inference Server client container with: `singularity run --fakeroot --nv -B ~/End_to_end_CV/workspace:/workspace triton_client.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace`

Then, open jupyter lab in browser: http://localhost:8888 and continue the lab by running `3.Model_deployment_with_Triton_Inference_Server.ipynb`

As soon as you are done with that, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl d` in the terminal window. 

You are now ready to run the DeepStream container: `singularity run --fakeroot --nv -B ~/End_to_end_CV/workspace:/opt/nvidia/deepstream/deepstream-6.1/workspace deepstream.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-6.1/workspace`

Again, open jupyter lab in browser: http://localhost:8888 and complete the material by running `4.Model_deployment_with_DeepStream.ipynb` and `5.Measure_object_size_using_OpenCV.ipynb`.

Congratulations, you've successfully built and deployed an end-to-end computer vision pipeline!

## Running using Docker

Run the material via a python virtual environment and Docker containers. Root privileges are required using `sudo`. If you don't have root privileges on your local system, please follow the above instructions on how to run the lab using Singularity.

### Installing the prerequisites

1. Install `docker-ce` by following the [official instructions](https://docs.docker.com/engine/install/). Once you have installed docker-ce, follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) to ensure that docker can be run without `sudo`.

2. Install `nvidia-container-toolkit` by following the [install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

3. Get an NGC account and API key:
    - Go to the [NGC](https://ngc.nvidia.com/) website and click on `Register for NGC`
    - Click on the `Continue` button where `NVIDIA Account (Use existing or create a new NVIDIA account)` is written
    - Fill in the required information and register, then proceed to log in with your new account credentials
    - In the top right corner, click on your username and select `Setup` in the dropdown menu
    - Proceed and click on the `Get API Key` button
    - Next, you will find a `Generate API Key` button in the upper right corner. After clicking on this button, a dialog box should appear and you have to click on the `Confirm` button
    - Finally, copy the generated API key and username and save them somewhere on your local system


4. Install NGC CLI
    - Log in with your account credentials at [NGC](https://ngc.nvidia.com/)
    - In the top right corner, click on your username and select `Setup` in the dropdown menu
    - Proceed and click on the `Downloads` button in the CLI panel
    - Select `AMD64 Linux` and follow the instuctions
    - Open the terminal on your local system and log in to the NGC docker registry (`nvcr.io`) using the command `docker login nvcr.io` and enter `$oauthtoken` as Username and your `API Key` as Password.  
 
       

### Install TAO Toolkit and dependencies

TAO Toolkit is a Python pip package that is hosted on the NVIDIA PyIndex. The package uses the docker restAPI under the hood to interact with the NGC Docker registry to pull and instantiate the underlying docker containers. You must have an NGC account and an API key associated with your account.

1. Create a new `conda` environment using `miniconda`:

    - Install `Miniconda` by following the [official instructions](https://conda.io/projects/conda/en/latest/user-guide/install/).
    - Once you have installed `miniconda`, create a new environment by setting the Python version to 3.6
    
        `conda create -n launcher python=3.6`
    
    - Activate the `conda` environment that you have just created
    
        `conda activate launcher`
    
    - When you are done with your session, you may deactivate your `conda` environment using the `deactivate` command
    
        `conda deactivate`
   

2. Install the TAO Launcher Python package called `nvidia-tao` into the conda launcher environment:
    
    `conda activate launcher`
    `pip3 install nvidia-tao`

3. Invoke the entrypoints using the this command `tao -h`. You should see the following output:
```
    usage: tao 
             {list,stop,info,augment,bpnet,classification,detectnet_v2,dssd,emotionnet,faster_rcnn,fpenet,gazenet,gesturenet,
             heartratenet,intent_slot_classification,lprnet,mask_rcnn,punctuation_and_capitalization,question_answering,
             retinanet,speech_to_text,ssd,text_classification,converter,token_classification,unet,yolo_v3,yolo_v4,yolo_v4_tiny}
             ...

    Launcher for TAO

    optional arguments:
    -h, --help            show this help message and exit

    tasks:
          {list,stop,info,augment,bpnet,classification,detectnet_v2,dssd,emotionnet,faster_rcnn,fpenet,gazenet,gesturenet,heartratenet
          ,intent_slot_classification,lprnet,mask_rcnn,punctuation_and_capitalization,question_answering,retinanet,speech_to_text,
          ssd,text_classification,converter,token_classification,unet,yolo_v3,yolo_v4,yolo_v4_tiny}
```

   For more info, visit the [TAO Toolkit documentation](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html).

4. Install other dependencies needed to run the lab:
    `pip install jupyterlab matplotlib fiftyone attrdict tqdm`
    `pip install nvidia-pyindex tritonclient[all]`

## Run the Lab

Activate the conda launcher environment:
    
    `conda activate launcher`
    
You are to run the first two notebooks `1.Data_labeling_and_preprocessing.ipynb` and `2.Object_detection_using_TAO_YOLOv4.ipynb` in the `launcher environment`

Launch the jupyter lab with:

`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=~/End_to_end_CV/workspace` 

Remember to set the `--notebook-dir` to the location where the `project folder` where this material is located.

Then, open jupyter lab in the browser at http://localhost:8888 and start working on the lab by clicking on the `Start_here.ipynb` notebook.

When you are done with `1.Data_labeling_and_preprocessing.ipynb` and `2.Object_detection_using_TAO_YOLOv4.ipynb`, shut down jupyter lab by selecting `File > Shut Down` in the top left corner and move to the next section in this guide.

### Run Triton Inference Server notebook 

To start the Triton Inference Server instance, you will need to run a container along with the `launcher`  virtual enviroment. This is to emulate the client-server mechanism but on the same system. To start the server, `open a new terminal` and launch the command:
```
docker run \
  --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/End_to_end_CV/workspace/models:/models \
  nvcr.io/nvidia/tritonserver:22.05-py3 \
  tritonserver \
  --model-repository=/models \
  --exit-on-error=false \
  --model-control-mode=poll \
  --repository-poll-secs 30
```
In order to work properly in this lab, the triton sever version should match the TAO Toolkit version that was installed (visible by running `tao info`). Containers with the same tag avoid version mismatches and conficts that may prevent you from running and deploying your models. The path to the local model repository needs to be set as well in order to be mapped inside the container.

After starting Triton Server, you will see an output on the terminal showing `the server starting up and loading models`. This implies Triton is ready to accept inference requests.
```
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| <model_name>         | <v>     | READY  |
| ..                   | .       | ..     |
| ..                   | .       | ..     |
+----------------------+---------+--------+
...
...
...
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

Now you can go back to your browser with jupyter lab open and run `3.Model_deployment_with_Triton_Inference_Server.ipynb`

When you are done with the notebook, shut down jupyter lab by selecting `File > Shut Down` as well as the Triton Docker container of the server by pressing `ctrl c` in the logs terminal. 

You are now ready to move on with the final notebooks.

### Run DeepStream and OpenCV notebooks

To run the DeepStream content, build a Docker container by following these steps:  

- Open a terminal window, navigate to the directory where `Dockerfile_deepstream` is located. 
- Run `sudo docker build -f Dockerfile_deepstream --network=host -t <imagename>:<tagnumber> .`, for instance: `sudo docker build -f Dockerfile_deepstream --network=host -t deepstream:1.0 .`
- Next, execute the command: `sudo docker run --rm -it --gpus=all -v ~/End_to_end_CV/workspace:/opt/nvidia/deepstream/deepstream-6.1/workspace --network=host -p 8888:8888 deepstream:1.0`

flags:
- `--rm` will delete the container when finished
- `-it` means run in interactive mode
- `--gpus` option makes GPUs accessible inside the container
- `-v` is used to mount host directories in the container filesystem
- `--network=host` will share the host’s network stack to the container
- `-p` flag explicitly maps a single port or range of ports

When you are inside the container, launch jupyter lab: 
`jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/opt/nvidia/deepstream/deepstream-6.1/workspace`. 

Open the browser at `http://localhost:8888` and start working on `4.Model_deployment_with_DeepStream.ipynb` notebook. Then, move to `5.Measure_object_size_using_OpenCV.ipynb` and complete the material.

As soon as you are done with that, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl d` in the terminal window.

Congratulations, you've successfully built and deployed an end-to-end computer vision pipeline!

## Known Issues

### tao
When installing the TAO Toolkit Launcher to your host machine’s native python3 as opposed to the recommended route of using a virtual environment, you may get an error saying that `tao binary wasn’t found`. This is because the path to your `tao` binary installed by pip wasn’t added to the `PATH` environment variable in your local machine. In this case, please run the following command:

`export PATH=$PATH:~/.local/bin`

### NGC

You can see an error message stating:

`ngc not found, did you mean: ......... `

You can resolve this by setting the path to ngc within the conda launcher environment as:

`echo "export PATH=\"\$PATH:$(pwd)\"" >> ~/.bash_profile && source ~/.bash_profile`
 
