# AloNet

<B>Author: Solam Lee (solam@yonsei.ac.kr)</B>
<br><br>


|AloNet Demo Video|
|---|
|[![AloNet Demo Video](https://img.youtube.com/vi/J3WQLAJ9iew/0.jpg)](https://www.youtube.com/watch?v=J3WQLAJ9iew)|
| Please click the above preview image for playing the demo video |


<br>
AloNet is a convolutional neural network based on U-Net, which can identify hair loss and scalp area by analyzing a clinical photograph. This model was developed for automated calculation of the Severity of Alopecia Tools score to assess of patients with alopecia areata.

This repository posts the program code and the relevant data used in the paper titled “Clinically Applicable Deep Learning Framework for Measuring the Extent of Hair Loss in Patients with Alopecia Areata.”

Along with the programs in the “/Program/” directory, a total of 2716 pixelwise annotations used to train the hair loss identifier (mask), and the hair loss identifier (target) can be find in the “/Data/” directory. However, please note that patients’ clinical photographs cannot be made publicly available because of strict privacy regulations.

Before using the AloNet program with your dataset, you should convert your dataset into numpy files. One clinical photograph (saved in .jpg with RGB format) is needed for each annotation for the scalp area (saved in .gif in black and white) and hair loss area (saved in .gif in black and white), respectively. Please make sure that they have the same image size or the conversion will fail.

We are currently working on several postprocessing algorithms for AloNet to be available for general use. The Flask web application and its code will be made available publicly when the program is ready to use.

For inquiries regarding institutional adoption of AloNet, co-work, or of any other nature, please contact the author.
Several samples can be found in the “/Samples/” directory, including the following:
<br><br>

![Sample](/Samples/sample1.png)
![Sample](/Samples/sample4.png)
![Sample](/Samples/sample7.png)
![Sample](/Samples/sample10.png)
![Sample](/Samples/sample40.png)



<br><Br>

<h2><B>Deep Learning Method</B></h2>
<br>

![Sample](/Method.jpg)
<br>

AloNet consists of two major components: 1) the hair loss identifier, which classifies each pixel as “hair loss” or not, and 2) the scalp area identifier, which classifies each pixel as “scalp area” or not. Although these two identifiers were trained with different annotated inputs, they shared most of the network configurations. The scalp area identifier was required for the following reasons: 1) Calculation of the Severity of Alopecia Tool score is based on the extent of hair loss in the total scalp area; both variables were required. However, it would be very cumbersome if the user needs to manually prepare and input a hand-drawn scalp area for every image. Therefore, we sought to develop an end-to-end framework that can automatically extract both the scalp area and the hair loss area simultaneously from a single image input. 2) The hair loss identifier achieved better performance when it received the input image after being masked with the predicted output of the scalp identifier. Although our earliest model in which no masking was used also showed fair performance (Jaccard index of 0.935 for identifying hair loss), the current model showed better performance than the prototype.

 When training the scalp identifier, the images were first divided into two subgroups according to the similarity to the morphological characteristics of the ground truth for the scalp area. The temporal subset consisted of the left- and right-view images, whereas the midline subset consisted of the top- and back-view images. Likewise, the scalp area identifier can be divided into two networks: one for training and inference for the temporal subsets and the other for the midline subsets. The hair loss identifier, on the other hand, was trained with all images to work for any inputs regardless of the direction.

 All images were resized to 320x320 pixels and loaded onto the network. The number of input channels was changed to three because of RGB (red, green, blue)-colored input images. The drop-outs at a rate of 0.25 for each layer were added in the upscale steps to avoid overfitting. The inputs from the previous layer were batch normalized, and the rectified linear unit was adopted as an activation function. The optimizer, initial learning rate, and gamma for the scheduler was Adam, 0.0005, and 0.1, respectively. Loss function was a cross-entropy loss with a class weight of 1:5 for the region of interest and the others, respectively. The output from the last layer was sigmoided so that the values can represent a confidence score (range 0 to 1) for each class label.
The process was conducted using a personal computer running Ubuntu version 19.04 (Canonical, London) and with PyTorch deep learning framework version 1.2.0 (https://pytorch.org/) with CUDA 10.1/cuDNN 7.6 dependencies for GPU acceleration (Nvidia, California). The system was equipped with an Intel i9 9960X 32-threaded 3.10-GHz CPU (Intel, California), 64 GB DDR4 RAM, 2 TB SSD, and 4 units NVIDIA GeForce RTX 2080Ti 11 GB GPU
