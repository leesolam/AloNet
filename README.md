# AloNet

<B>Author: Solam Lee, MD (solam@yonsei.ac.kr)</B>

This repository posts the program code and the relevant data used in the paper titled "Computer-Aided Measurement of the Severity of Alopecia Tool by Using Convolutional Neural Network for Hair Loss Segmentation".

A total of 2916 pixelwise annotations used for train the hair loss identifier (mask) and the hair loss identifier (target) could be find in the "/Data/" directory. However, please note that the clinical images could not be made publicly available because of strict privacy regulation.

To use AloNet program with your dataset, you should convert your dataset into numpy files. The clinical image (.jpg in RGB format) need one annotation for the scalp area (.gif) and the hair loss (.gif), respectively. Please note that they should have same image size each other, or the converion will fail.

We are now currently working on several postprocessing algorithms for AloNet to be available for general use. The Flask web application and its code will be made available publicly when the program is ready to use.

For any inquiry on an adoption of AloNet for the instituion, co-work, or others, contact the e-mail mentioned above.
