# AloNet

<B>Author: Solam Lee (solam@yonsei.ac.kr)</B>

AloNet is a convolutional neural network based on U-Net that can identify the hair loss and the scalp area by analying clinical photograph. This model was developed for the automated calculation of the Severity of Alopecia Tools (SALT) score in assessment of patients with alopecia areata.

This repository posts the program code and the relevant data used in the paper titled "Computer-Aided Measurement of the Severity of Alopecia Tool by Using Convolutional Neural Network for Hair Loss Segmentation".

Along with the programs in the "/Program/" directory, a total of 2716 pixelwise annotations used for train the hair loss identifier (mask) and the hair loss identifier (target) could be find in the "/Data/" directory. However, please note that the clinical photograph of the patients could not be made publicly available because of strict privacy regulation.

To use AloNet program with your dataset, you should convert your dataset into numpy files in the first. One clinical photograph (saved in .jpg with RGB format) need one annotation for the scalp area (saved in .gif with black&white color) and the hair loss (saved in .gif with black&white color), respectively. Please make sure that they have same image size each other, or the conversion will fail.

We are now currently working on several postprocessing algorithms for AloNet to be available for general use. The Flask web application and its code will be made available publicly when the program is ready to use.

For inquiry on an institutinal adoption of AloNet, co-work, or any others, please contact the author.

Some samples can be found in the "/Samples/" directory, including the followings:

![Sample](/Samples/sample1.png)
![Sample](/Samples/sample4.png)
![Sample](/Samples/sample7.png)
![Sample](/Samples/sample10.png)
![Sample](/Samples/sample40.png)
