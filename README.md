## Synthesization of documents for object detection task

#### Motivation


In a lot of document object detection tasks such as signature detection, card detection, 
large volume of data are required by complex CV algorithms. Manual annotation of objects can take up a lot of time, synthesizing document images is a handy way to get around this problem. This repo will show how to insert objects into a document template and 
at the same time, automatically generate the bounding box, key points etc. annoation in COCO format for the modelling task.

For illustration purpose, I will demostrate how to insert signature .

![Signature](https://github.com/JasonSCFu/Document-Data-Synthesization/blob/main/debug/idcard_train/01_054.png)

into a document
![Document](https://github.com/JasonSCFu/Document-Data-Synthesization/blob/main/source/det_background_images/de_val_13.jpg)

The output will look something like:
![Result](https://github.com/JasonSCFu/Document-Data-Synthesization/blob/main/debug/idcard_det_train/0_16725821071229827.jpg)




#### Steps to generate the above document:

> 1: Place document template in folder source\det_background_images, 
place signature image in debug\idcard_train.

> 2: Run sig_doc_generation.py, the output will be saved in debug\idcard_det_train

> 3: Run convert2coco.py, the coco format annotation will be saved in debug\card_coco\test


I downloaded the signatures data from [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset?resource=download)

Background documents are from [Microsoft Research Asia Document AI](https://github.com/doc-analysis/XFUND)
