# AloNet

<B>Author: Solam Lee (solam@yonsei.ac.kr)</B>

AloNet is a convolutional neural network based on U-Net that can identify the hair loss and the scalp area by analying clinical photograph. This model was developed for the automated calculation of the Severity of Alopecia Tools (SALT) score in assessment of patietns with alopecia areata.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png


This repository posts the program code and the relevant data used in the paper titled "Computer-Aided Measurement of the Severity of Alopecia Tool by Using Convolutional Neural Network for Hair Loss Segmentation".

Along with the programs in the "/Program/" directory, a total of 2716 pixelwise annotations used for train the hair loss identifier (mask) and the hair loss identifier (target) could be find in the "/Data/" directory. However, please note that the clinical images could not be made publicly available because of strict privacy regulation.

To use AloNet program with your dataset, you should convert your dataset into numpy files in the first. One clinical photograph (saved in .jpg with RGB format) need one annotation for the scalp area (saved in .gif with black&white color) and the hair loss (saved in .gif with black&white color), respectively. Please note that they should have same image size each other, or the conversion will fail.

We are now currently working on several postprocessing algorithms for AloNet to be available for general use. The Flask web application and its code will be made available publicly when the program is ready to use.

For any inquiry on an adoption of AloNet for the instituion, co-work, or others, please contact the author.
