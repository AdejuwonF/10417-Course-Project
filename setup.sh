#!/bin/bash

mkdir COCO_DATASET
cd COCO_DATASET
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
chmod 777 val2017.zip
chmod 777 annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip