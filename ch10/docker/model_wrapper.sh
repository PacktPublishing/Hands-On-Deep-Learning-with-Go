#!/bin/bash

set -e

echo bucket: s3://$BUCKET_NAME
echo ""

pwd
cd /Hands-on-Deep-Learning-in-Go/ch10
go run main.go
echo "Performed inference on CIFAR CNN Model!"
