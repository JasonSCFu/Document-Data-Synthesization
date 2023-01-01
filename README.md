## Synthesization of documents for object detection task

#### Motivation


In a lot of document object detection tasks such as signature detection, card detection, 
large volume of data are required by complex CV algorithms. Manual annotation of objects can take up a lot of time, synthesizing document images is a handy way to get around this problem. This repo will show how to insert objects into a document template and 
at the same time, automatically generate the bounding box, key points for the modelling task.

For illustration purpose, I will demostrate how to insert signature .

![Signature](https://github.com/JasonSCFu/Document-Data-Synthesization/blob/main/debug/idcard_train/01_027.png)

into a document
![Document](https://github.com/JasonSCFu/Document-Data-Synthesization/blob/main/source/det_background_images/de_val_13.jpg)

The output will look something like:
![Result]()




#### Steps:

> 1: place document template in source\det_background_images
place signature in debug\idcard_train

> 2: run sig_doc_generation.py, the output will be saved in debug\idcard_det_train

> 3: run convert2coco.py, the output will be saved in debug\card_coco\test


I download the English signatures data from [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset?resource=download)

Background documents are from [Microsoft Research Asia Document AI](https://github.com/doc-analysis/XFUND)
