# AloNet

<B>Author: Solam Lee (solam@yonsei.ac.kr)</B>

AloNet is a convolutional neural network based on U-Net, which can identify hair loss and scalp area by analyzing a clinical photograph. This model was developed for automated calculation of the Severity of Alopecia Tools score to assess of patients with alopecia areata.

This repository posts the program code and the relevant data used in the paper titled “Clinically Applicable Deep Learning Framework for Measuring the Extent of Hair Loss in Patients with Alopecia Areata.”

Along with the programs in the “/Program/” directory, a total of 2716 pixelwise annotations used to train the hair loss identifier (mask), and the hair loss identifier (target) can be find in the “/Data/” directory. However, please note that patients’ clinical photographs cannot be made publicly available because of strict privacy regulations.

Before using the AloNet program with your dataset, you should convert your dataset into numpy files. One clinical photograph (saved in .jpg with RGB format) is needed for each annotation for the scalp area (saved in .gif in black and white) and hair loss area (saved in .gif in black and white), respectively. Please make sure that they have the same image size or the conversion will fail.

We are currently working on several postprocessing algorithms for AloNet to be available for general use. The Flask web application and its code will be made available publicly when the program is ready to use.

For inquiries regarding institutional adoption of AloNet, co-work, or of any other nature, please contact the author.
Several samples can be found in the “/Samples/” directory, including the following:


![Sample](/Samples/sample1.png)
![Sample](/Samples/sample4.png)
![Sample](/Samples/sample7.png)
![Sample](/Samples/sample10.png)
![Sample](/Samples/sample40.png)
