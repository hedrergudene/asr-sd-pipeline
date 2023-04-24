#!/bin/bash
az ml compute create -f ./create-cpu-cluster.yaml
az ml compute create -f ./create-gpu-cluster.yaml