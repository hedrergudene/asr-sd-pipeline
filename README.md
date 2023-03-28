# Towards industrial-level Conversational-AI solutions

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Description](#description)
- [3. Training data](#training-data)
- [4. Components](#components)
- [5. IAM](#iam)
- [6. Quickstart](#quickstart)
- [7. Call for contributions](#call-for-contributions)
- [8. License](#license)
---


## Introduction

An Azure Machine Learning (AML) pipeline is a workflow that can be run independently of a complete machine learning task. An AML pipeline helps <mark>standardize best practices for producing a machine learning model</mark>, enables the team to run at scale, and improves model building efficiency.

The core of a machine learning pipeline is breaking an entire machine learning task into a multi-step workflow. Each step is a manageable component that can be individually developed, optimized, configured, and automated. The steps are connected through well-defined interfaces. The AML pipeline service automatically organizes all dependencies between the pipeline steps. This modular approach brings two key benefits:

* Standardize the practice of machine learning operation (MLOps) and support scalable team collaboration.
* Efficiency in training and cost reduction.

In this use case, we provide an easy to adapt and scalable approach to implement a tabular ML solution.


## Description

The following diagram shows a detailed structure of how a simple ML workflow looks like:

<img src="images/project-outlook.PNG"  width="100%" height="100%" style="display: block; margin: 0 auto">

There are several components to be discussed in order to keep track of every step through the workflow:

1. Code is stored in a repository, and unit test should be performed every time a new change occurs.
2. Data is stored in a container, and every time the training pipeline is triggered, it is fetched by a component to create an **`MLTable` checkpoint** (see next section).
3. Model training and evaluation is performed using stratified K-Fold strategy, inside a **bayesian hyperparameter optimisation** procedure.
4. Loss function, evaluation metrics and results are monitored using [**MLFlow**](https://mlflow.org/). Instead of using [hyperparameter tuning functionalities in AzureML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters), a low-level solution provided by [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) is recommended.
5. A checkpoint in model registry is created with the best hyperparameter configuration, ready to be deployed.


## Training data

### Data Assets (`MLTable`)

An ideal resource for storing, managing and distributing data are [Data Assets](https://learn.microsoft.com/en-us/azure/machine-learning/concept-data). These allow to:

* Create an object that is robust to changes (for example, a column name changes): All consumers of the data must independently update their Python code. Other examples can involve type changes, added / removed columns, encoding change, etc.
* The data size increases - If the data becomes too large for Pandas to process, all the consumers of the data will need to switch to a more scalable library (PySpark/Dask).

Azure Machine Learning Tables allow the data asset creator to define the materialization blueprint in a single file. Then, consumers can then easily materialize the data into a data frame. The consumers can avoid the need to write their own Python parsing logic. The creator of the data asset defines an [`MLTable`](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-mltable) file.

We recommend co-location of the `MLTable` file with the underlying data, for example:

```
  ├── <container-name>
  │   ├── MLTable
  │   ├── file_1.csv
  .
  .
  .
  │   ├── file_n.csv
```

Co-location of the `MLTable` file with the data ensures a self-contained artifact that stores all needed resources in that one folder, regardless of whether that folder is stored on your local drive, in your cloud store, or on a public http server. Plus, real-life environments are not static, therefore this practice ensures that datacheckpoints are always up to date.

Since the `MLTable` will co-locate with the data, the paths defined in the `MLTable` file should be relative to the location of the `MLTable` file. In the `./setup` folder, you will find the `smoke_detection_iot.csv` file, which has been used for testing purposes.

### Assumptions

This use case is prepared for supervised learning problems, where the following assumptions are made over the data:

* Dataset only contain numerical features; categorical ones must be preprocessed in advanced (see `call-for-contributions` section).
* Preprocessing performed for features and target variable (if it is a regression scenario) is a normal standarisation. This might change depending on your specific domain, where other possible transformations are Tweedie and binary crossentropy losses for highly-concentrated distributions.


## Components

### Nomenclature

Each of the pipeline components is executed within an *AML* computing cluster. These are determined by the **virtual machines** that host them, and whose nomenclature has the following meaning:

* `STANDARD`: *Tier* recommended for VM availability purposes.
* `D`: VMs for any purpose.
* `L`: Optimized in terms of storage.
* `S`: Provides *premium* storage and offers a local SSD for *cache*.
* `M`: Optimized in terms of memory.
* `G`: Optimized in terms of memory and storage.
* `E`: Optimized for *multi-thread* operations in memory.
* `V`: Optimized for intensive graphics work and remote viewing *workloads*.
* `C`: Optimized for high performance computing and ML *workloads*.
* `N`: Available GPU(s).

In order to optimize the resources we use in each module of our process, the following virtual machines are recommended:

* `cpu-cluster`: The default host is `STANDARD_DS11_v2`. In case four processing *cores* are needed, consider using `STANDARD_DS3_v2`, and for CPU-intensive *Machine Learning* training, it is recommended to choose `Standard_DC16ds_v3`.
* `gpu-cluster`:
  * For training on a single GPU, and in order to test solutions and proofs of concept, it is recommended to use (with small *batch* size) `Standard_NC24ads_A100_v4` because of its cheap price.
  * For single GPU training on stable solutions and validated pipelines, it is preferable to use one like `Standard_NC16as_T4_v3` if the model does not require large memory capacity, or `Standard_NC24ads_A100_v4` for its processing power.
  * For training on multiple GPUs, and in order to test solutions and proofs of concept, it is recommended to use (with small *batch* size) `Standard_NC12` due to its cheap price, when making Tesla K80 GPUs available.
  * For training on multiple GPUs in stable solutions and validated pipelines, it is preferable to use one like `Standard_NC64as_T4_v3` (4GPUs, Tesla T4) if the model does not require large memory capacity, `Standard_NC48ads_A100_v4` (2GPUs, Tesla A100) for its good price-performance ratio, or `Standard_ND96amsr_A100_v4` (8GPUs, Tesla A100) for its extensive processing power.

In the `./setup` folder, you will find a shell script (together with configuration files) to create CPU and GPU clusters.


### Structure

Pipelines, and in particular  AML components, are based on those of [*Kubeflow*](https://www.kubeflow.org/docs/components/pipelines/v1/sdk-v2/), so that the reader will be familiaried with the structure of these projects if he has previously used them. We recommend, in order to maintain good practices and avoid possible errors, to maintain the standardised format that is followed in the components of this project, and to freely manipulate the part of them where the code is entered, conveniently adjusting the rest of the files.

For components that require only CPU units, we follow a structure like this:

```
  ├──<component-name>              # Where component lies
  │   ├── .amlignore               # Data not to be included in AML component creation
  │   ├── conda.yaml               # Conda environment and dependencies definition
  │   ├── <component-name>.yaml    # Component configuration file
  │   │                            
  │   └── main.py                  # Entrypoint

```

For those components that require GPU acceleration, the corresponding *script* that launches the *job* with the pipeline that contains it is prepared to automatically detect what type of virtual machine is being used, and suitably adapt the configuration file associated with these components. All available VMs in your timezone can be collected from the following [link](https://azureprice.net/).

Now let's see how a component with GPU(s) support is organized:

```
  ├──<component-name>                         
  │   ├── config                              
  │   │   ├── multi_gpu_config.yaml           # Accelerate config for DistributedDataParallel
  │   │   └── single_gpu_config.yaml          # Accelerate config for single-GPU training
  │   ├── docker      
  │   │   └── Dockerfile                      # File with base image and dependencies (AML will do everything else)
  │   ├── src                                 # Scripts containing additional functionalities
  │   │   ├── dataset.py
  │   │   ├── fitter.py
  │   │   ├── model.py  
  │   │   ...
  │   │   └── utils.py
  │   ├── .amlignore                             
  │   ├── <component-name>.yaml               # Single-GPU component definition
  │   ├── <component-name>_distributed.yaml   # Multi-GPU component definition
  │   └── main.py                             # Entrypoint
  
```

For potential integrations with large-scale training, requiring specific use of optimizers such as [*DeepSpeed*](https://www.deepspeed.ai/) or other types of parallelization such as *DistributedTensorParallel*, please request assistance from the contributors of this repo, to adapt both the configuration files and the virtual execution environment to your needs.

### Register

Hosting your code in a repository is the best way to track changes and serialise releases. However, there are additional functionalities within the AML ecosystem that allow users to *plug and play* with pipelines. By opening a terminal and running the shell script contained in `./setup` folder, with `<workspace-name>` and `<resource-group-name>` as parameters, an instance of your custom components will be created, so that you can prepare, run and monitor pipeline workflows using only the UI (in the `Pipelines` section). 

## IAM

AML computing clusters will use a service account to which we must assign a series of roles in order to execute these processes successfully:
* Storage Blob Data Contributor (in storage account resource)
* Storage Queue Data Contributor (in storage account resource)
* AzureML Data Scientist (in AML resource)
* Access to [KeyVaults](https://learn.microsoft.com/en-us/azure/key-vault/general/assign-access-policy?tabs=azure-portal)


## Quickstart

Once the environment has been created, permissions for service account have been granted and you filled the configuration file with your own data, the fastest way to run AML pipelines is by opening a terminal and launching `azureml_pipeline.py` script, using as parameter the configuration file path.


## Call for contributions

Despite including and end-to-end solution to model design in AML, the following additional features are expected to be developed:

- [X] Adapt `build_MLTable` function in pipeline job script to replicate Data Asset config file (done by @demstalferez).
- [ ] Make preprocessing flexible enough to ingest categorical features, and either perform label or one-hot encoding.
- [ ] `RAPIDS` integration to train ML models using GPU.
- [ ] Inference endpoint creation.
- [ ] User-friendly app development.
- [ ] Live application by using a real IoT device.


## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).