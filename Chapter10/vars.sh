#!/bin/bash


# AWS vars
export BUCKET_NAME="hodlgo-models"
export MASTERTYPE="m3.medium"
export SLAVETYPE="t2.medium"
export SLAVES="2"
export ZONE="ap-southeast-2b"

# K8s vars
export NAME="hodlgo.k8s.local"
export KOPS_STATE_STORE="s3://hodlgo-cluster"
export PROJECT="hodlgo"
export CLUSTER_NAME=$PROJECT


# Docker vars
export VERSION_TAG="0.1"
export MODEL_CONTAINER="hodlgo-model"
