#!/bin/bash
az ml compute create -f ./create-cpu-cluster.yaml
az ml compute create -f ./create-gpu-cluster-t4.yaml
az ml compute create -f ./create-gpu-cluster-a100.yaml