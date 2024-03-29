Contributing
------------

Please use the following guidelines when contributing to this project. 

Before contributing significant changes, please begin a discussion of the desired changes via a GitHub Issue to prevent doing unnecessary or overlapping work.

## License

The preferred license for source code contributed to this project is the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) and for documentation, including Jupyter notebooks and text documentation, is the Creative Commons Attribution 4.0 International (CC BY 4.0) (https://creativecommons.org/licenses/by/4.0/). Contributions under other, compatible licenses will be considered on a case-by-case basis.

## Styling

Please use the following style guidelines when making contributions.

### Source Code
* Two-space indentation, no tabs
* To the extent possible, variable names should be descriptive
* Code should be documentation with detail like what function does and returns making the code readable. The code should also have proper license at the beginning of the file.
* The following file extensions should be used appropriately:
	* Python = .py

### Jupyter Notebooks & Markdown
* When they appear inline with the text; directive names, clauses, function or subroutine names, variable names, file names, commands and command-line arguments should appear between two backticks.
* Code blocks should begin with three backticks to enable appropriate source formatting and end with three backticks.
* Leave an empty line before and after the codeblock.
Emphasis, including quotes made for emphasis and introduction of new terms should be highlighted between a single pair of asterisks
* A level 1 heading should appear at the top of the notebook as the title of the notebook.
* A horizontal rule should appear between sections that begin with a level 2 heading.


## Contributing Labs/Modules
#### DeepStream Triton Inference Server Integration 
* In the existing end-to-end CV repo, only models build from TAO or optimized by TRT can be deployed on DeepStream (streaming video). DeepStream Triton Inference Server Integration enables the use of trained model from desired framework, such as TensorFlow, TensorRT, PyTorch, or ONNX-Runtime, and directly run inferences on streaming video.
	* Task 1:  Extend end-to-end CV repo with DeepStream Triton Inference Server Integration
	* Task 2:  Upgrade end-to-end CV repo to TAO Toolkit 4.0.1 and Add AutoML section. AutoML is a TAO Toolkit API service that automatically selects     		           deep learning hyperparameters for a chosen model and dataset.
#### Body Pose Estimation
* The use for Body pose estimation in CV domain include
	* Tracking customer who picked or dropped products in a retail store[real-time inventory]
	* Track the safety of factory personnel
 	* E-health monitoring system
* Task: Create an end-to-end pose body estimation material (dataset prep., TAO Train, and DeepStream deployment) 


### Directory stucture for Github

Before starting to work on new lab it is important to follow the recommended git structure as shown below to avoid reformatting.

Each lab will have following files/directories consisting of training material for the lab.
* jupyter_notebook folder: Consists of jupyter notebooks and its corresponding images.  
* source_code folder: Source codes are stored in a separate directory because sometime not all clusters may support jupyter notebooks. During such bootcamps, we should be able to use the source codes directly from this directory. 
* presentations: Consists of presentations for the labs ( pdf format is preferred )
* Dockerfile and Singularity: Each lab should have both Docker and Singularity recipes.
 
The lab optionally may also add custom license in case of any deviation from the top level directory license ( Apache 2.0 ). 


### Git Branching

Adding a new feature/lab will follow a forking workflow. Which means a feature branch development will happen on a forked repo which later gets merged into our original project (GPUHackathons.org) repository.

![Git Branching Workflow](workspace/jupyter_notebook/images/git_branching.jpg)

The 5 main steps depicted in image above are as follows:
1. Fork: To create a new lab/feature the GPUHackathons.org repository must be forked. Fork will create a snapshot of GPUHackathons.org repository at the time it was forked. Any new feature/lab that will be developed should be based on the develop branch of the repository.
2.  Clone: Developer can than clone this new repository to local machine
Create Feature Branch: Create a new branch with a feature name in which your changes will be done. Recommend naming convention of feature branch is naming convention for branch: ende2end-cv-<feature_name>. The new changes that developer makes can be added, committed and pushed
3. Push: After the changes are committed, the developer pushes the changes to the remote branch. Push command helps the local changes to github repository
4. Pull: Submit a pull request. Upon receiving pull request a Hackathon team reviewer/owner will review the changes and upon accepting it can be merged into the develop branch of GpuHacakthons.org

Git Branch details are as follows:

* master branch: Consists of the stable branch. 
	* origin/master to be the main branch where the source code of HEAD always reflects a production-ready state
	* Merge request is possible through:  develop branch
* develop branch: branched from master branch
	* Must branch from: master branch
	* Must merge back into: master branch
	* It is the main development branch where the source code of HEAD always reflects a state with the latest delivered development changes for the next release.
	* When the source code in the develop branch reaches a stable point and is ready to be released, all of the changes should be merged back into master somehow and then tagged with a release number
	* All feature development should happen by forking GPUHackathons.org and branching from develop branch only.
