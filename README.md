# Towards industrial-level Multi-Speaker Speech Recognition solutions

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Description](#description)
- [3. Components](#components)
- [4. IAM](#iam)
- [5. Quickstart](#quickstart)
- [6. Call for contributions](#call-for-contributions)
- [7. License](#license)
---


## Introduction

In recent years, speech recognition technology has become ubiquitous in our daily lives, powering virtual assistants, smart home devices, and other voice-enabled applications. However, building a robust speech recognition system is a complex task that requires sophisticated algorithms and models to handle the challenges of different accents, background noise, and multiple speakers.

In this repository, we aim to provide a comprehensive overview of the latest advancements in speech recognition and speaker diarization using deep learning techniques. We will explore the underlying technologies, including neural networks and their variants, and provide code examples and tutorials to help developers and researchers get started with building their own speech recognition and speaker diarization systems by making minimal changes to the Azure Pipelines implementation provided.


## Description

Speech recognition is the task of automatically transcribing spoken language into text. It involves developing algorithms and models that can analyze audio recordings and identify the words and phrases spoken by a user. In recent years, deep learning models have shown great success in improving speech recognition accuracy, making it a hot topic in the field of machine learning.

Autoregressive models such as [Whisper](https://openai.com/research/whisper) provide excepcional transcriptions when combined with some additional preprocessing features, that we have picked up from [this excellent repo](https://github.com/guillaumekln/faster-whisper), being the quality of the timestamps returned for each audio segment rather poor. In this direction, phoneme-based speech recognition tools like [wav2vec2](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) handle timestamps perfectly, as these are finetuned to recognise the smallest unit of speech distinguishing one word from another.

A technique that makes both ends meet is [forced alignment](https://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Forced_alignment#:~:text=Forced%20alignment%20refers%20to%20the,automatically%20generate%20phone%20level%20segmentation.); a good introduction to this topic can be found [here](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html), and our implementation relies on [NeMo repo](https://nvidia.github.io/NeMo/blogs/2023/2023-08-nfa/).

<img src="images/sphx_glr_forced_alignment_tutorial_005.png"  width="70%" height="70%" style="display: block; margin: 0 auto">

Alignment can only be performed, however, when both model's vocabulary is matched. To overcome this issue, most recent approaches such as Dynamyc Time Warping (DTW) algorithm are applied to cross-attention weights to directly handle Whisper features to enhance word-level timestamps.

<img src="images/special_char.PNG"  width="50%" height="50%" style="display: block; margin: 0 auto">

Speaker diarization, on the other hand, is the process of separating multiple speakers in an audio recording and assigning each speaker to their respective segments. It involves analyzing the audio signal to identify the unique characteristics of each speaker, such as their voice, intonation, and speaking style. Speaker diarization is essential in applications such as call center analytics, meeting transcription, and language learning, where it is necessary to distinguish between different speakers in a conversation.

In this direction, [Multi-scale systems](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/) have emerged as a feasible solution to overcome traditional problems attached to time window selection. In our work, we make use of previous steps to enhance and optimise diarization runtime by providing segments VAD and accurate, word-level timestamps, which is particularly relevant in long audios.

Finally, in order to enhance natural language transcriptions quality and readability, a punctuation-based sentence alignment strategy has been implemented after both ASR and diarization steps.

<img src="images/summary.PNG"  width="100%" height="70%" style="display: block; margin: 0 auto">

When it comes to scalability, parallelisation plays a pivotal role. In this direction, we adopted `parallel_job` solution included in the *AML* toolkit, that allows to distribute your input across a defined number of devices asynchronously to speedup inference. Notice that there is not an immediate extrapolation of the code from standard pipelines to parallel components.

<img src="https://learn.microsoft.com/en-us/azure/machine-learning/media/how-to-use-parallel-job-in-pipeline/how-entry-script-works-in-parallel-job.png?view=azureml-api-2#lightbox"  width="60%" height="60%" style="display: block; margin: 0 auto">



## Structure

This service's main cornerstones are scalability, robustness and ease to deploy. In this direction, an API interface is provided to easily request batch processing jobs. While most of the parameters have default options, configuration related to storage paths, noSQL database credentials and secrets is required.


## Setup

## Storage
Input and output containers must be defined as AzureML Datastores. The reason behind is that we manage intermediate data to not generate an excessive amount of residual files, leading to greater costs; this is particularly relevant due to the fact that processed audios are one of those files. It can be achieved following [this steps](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-datastore?view=azureml-api-2&tabs=sdk-identity-based-access%2Csdk-adls-identity-access%2Csdk-azfiles-accountkey%2Csdk-adlsgen1-identity-access%2Csdk-onelake-identity-access).

## Tracking
Ideally, input and output blob paths inside those containers should vary, or processed data should be moved/deleted after each job. If this task is not handled, it would not raise any errors nor processing duplications, as we register every `unique_id` in a cosmosDB database, but potentially many unnecessary requests to that cosmosDB database and job inputs to each component will be ingested, leading to a suboptimal performance of your services.

## Keyvaults
An extensive usage of Azure Keyvault resource is made throughout the process. To be more precise, an asymmetric encription protocol (PGP) is used to ensure that anyone can encrypt data, but a limited number of profiles can decrypt it:

* `pubk_secret`: Holds the secret to the public key.
* `pk_secret`: Holds the secret to the private key.
* `pk_pass_secret`: Holds the secret to the private key's password.

These last two, however, are disabled; i.e., they must be enabled beforehand to access the secret. To that end, a service principal account credentials are also stored as secrets in the same keyvault, including (respectively) tenant, client and passwords under the identifiers:

* `secret_tenant_sp`
* `secret_client_sp`
* `secret_sp`

## IAM
AML computing clusters, together with AzuremL endpoint, will use a service account to which we must assign a series of roles in order to execute these processes successfully:
* Storage Blob Data Contributor (in storage account resource)
* Storage Queue Data Contributor (in storage account resource)
* AzureML Data Scientist (in AML resource)
* Access to [KeyVaults](https://learn.microsoft.com/en-us/azure/key-vault/general/assign-access-policy?tabs=azure-portal)
* Read and write (documents) roles in cosmosDB resource. Database and collection creation is not necessary (see `Setup`).


## Quickstart

Once the environment has been created, permissions for service account have been granted and you filled the configuration file with your own data, the fastest way to run AML pipelines is by opening a terminal and running the provided script to start an AzureML job:

```
cd stt_aml_deploy
python online_endpoint.py --config_path ./config/online_endpoint.yaml
```


## Call for contributions

Despite including and end-to-end solution to model design in AML, the following additional features are expected to be developed:

- [X] Speed up diarization step by using aligned ASR output.
- [X] Include CTranslate2 engine in ASR components.
- [X] Improve preprocessing techniques in an individual component to enhance stability.
- [X] Parallelise processing using distributed, asynchronous clusters.
- [X] Serialise pipeline implementation to avoid Microsoft bugs on `parallel_run_function`.
- [X] End-to-end, monolitic implementation using keyvaults and security protocols.
- [X] Enhance benchmark logging and CUDA capabilities checking.
- [X] API batch service deployment.
- [ ] Make sentence alignment more sensitive to short texts.


## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).