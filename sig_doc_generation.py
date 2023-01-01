from msilib.schema import Binary
import os
import random
import cv2
import json
import argparse
import threading 
import time
import imgaug as ia 
import numpy as np
from glob import glob 
from PIL import ImageFont, ImageDraw, Image
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from scipy import rand
random.seed(666)

class DetectImageGenerator():


    def __init__(self, background_image_path_list, hkidcard_image_path_list):
        self.background_image_paths = background_image_path_list
        self.hkidcard_image_paths = hkidcard_image_path_list
        ia.seed(666)

    ### change_background is to generate bbox in json and change to doc background
    def change_background(self, foreground_image, corner, output_image_path = "debug\cc.jpg"):
        
        image_h, image_w, _ = foreground_image.shape

        background_path = self.background_image_paths[random.randint(0,len(self.background_image_paths)-1)]

        background_image = cv2.imread(background_path)
        background_image = cv2.resize(background_image,(image_w,image_h))


        mask_fg = np.zeros(foreground_image.shape, dtype = np.uint8)  ### create black
        mask_bg = np.ones(background_image.shape, dtype = np.uint8) ### 

        
        cv2.fillConvexPoly(mask_fg,corner,(1,1,1)) 
        cv2.fillConvexPoly(mask_bg,corner,(0,0,0))  

        image = mask_fg*foreground_image + mask_bg*background_image

        width = image.shape[1]
        height = image.shape[0]

        cv2.imwrite(output_image_path,image)
        corner = corner.tolist()
        box4 = [corner[0],corner[4],corner[8],corner[12]]
        x1 = min(box4[0][0],box4[1][0],box4[2][0],box4[3][0])
        y1 = min(box4[0][1],box4[1][1],box4[2][1],box4[3][1])
        x2 = max(box4[0][0],box4[1][0],box4[2][0],box4[3][0])
        y2 = max(box4[0][1],box4[1][1],box4[2][1],box4[3][1])
        box2 = [[x1,y1],[x2,y2]]

        with open(output_image_path.replace(".jpg",".json").replace(".png",".json"),"w",encoding="utf-8") as f:
            json_label = {"file_name":output_image_path,"width":width,"height":height,"keypoints":[corner],"box2":[box2],"box4":[box4]}
            json_label = json.dumps(json_label,ensure_ascii = False)
            f.write(json_label)
        
        return image, corner

    def merge_img(self,input_img, background_image,contour,draw_position):

        '''
        :param contour: signature's 4 corner points, numpy format[(x1,y2),(x2,y2),...,] 
        :param draw_position: (x,y): control where to put the signature

        '''
        #if (input_img.shape[2] != background_image.shape[2]):
            #print("input_img shape != background_img shape")
            #return

        input_img_h = input_img.shape[0]
        input_img_w = input_img.shape[1]
        background_img_h = background_image.shape[0]
        background_img_w = background_image.shape[1]

        #if (input_img_h > background_img_h or input_img_w > background_img_w):
            #print("input_img size > background_img size")
            #return

        #if (((draw_position[0] + input_img_w)>background_img_w) or ((draw_position[1]+input_img_h)>background_img_h)):
            #print("draw_position + input_img > background_img range")
            #return

        output_img = background_image.copy()
        input_roi = output_img[draw_position[1]:draw_position[1]+input_img_h,draw_position[0]:draw_position[0]+input_img_w]
        img_mask = np.zeros((input_img_h,input_img_w,input_img.shape[2]),dtype = np.uint8)

        triangles_list = [contour]

        cv2.fillPoly(img_mask,triangles_list,color=(1,1,1))
        cv2.fillPoly(input_roi, triangles_list,color=(0,0,0))

        img_mask = img_mask*input_img
        output_ori = input_roi + img_mask
        output_img[draw_position[1]:draw_position[1]+input_img_h, draw_position[0]:draw_position[0]+input_img_w] = output_ori


        return output_img

    def key_aug(self,image,kps):
        kpsoi = KeypointsOnImage(kps, shape = image.shape)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        Affine_sometimes = lambda aug: iaa.Sometimes(0.5,aug)
        seq = iaa.Sequential(
            [
                sometimes(iaa.Crop(percent = (0,0.2))),
                Affine_sometimes(iaa.Affine(
                    scale = {"x": (0.9,1.1), "y": (0.9,1.1)},
                    translate_percent = {"x": (-0.2,0.2), "y": (-0.2,0.2)},
                    rotate = (-20,20),
                    shear = (-16,16),
                    order = [0,1],
                    cval = (0,255),
                    mode = ia.ALL


                )),

                iaa.SomeOf((0,2),
                    [
                        sometimes(
                            iaa.Superpixels(
                                p_replace = (0,1.0),
                                n_segments = (20,200)
                            )
                        ),

                        iaa.OneOf(
                            [
                                iaa.GaussianBlur((0,3)),
                                iaa.AverageBlur(k = (2,7)),
                                iaa.MedianBlur(k = (3,11)),

                            ]
                        ),
                    
                        iaa.Sharpen(alpha = (0,1),lightness = (0.75, 1.5)),

                        iaa.Emboss(alpha=(0,1),strength = (0,2)),

                        sometimes(iaa.OneOf(
                            [
                                iaa.EdgeDetect(alpha = (0,0.7)),
                                iaa.DirectedEdgeDetect(alpha = (0,0.7), direction=(0,1)
                                ),
                            ]
                        )),
                    
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0,0.05 * 255), per_channel = 0.5),

                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel = 0.5),
                        iaa.CoarseDropout((0.03,0.15),size_percent = (0.02,0.05), per_channel = 0.2
                        ),
                    ]),
                    iaa.Invert(0.05, per_channel = True),
                    iaa.Add((-10,10),per_channel = 0.5),
                    iaa.Multiply((0.5,1.5),per_channel = 0.5),
                    iaa.LinearContrast((0.5,2), per_channel = 0.5),


                    sometimes(iaa.ElasticTransformation(alpha = (0.5,3.5), sigma = 0.25)),

                    sometimes(iaa.PiecewiseAffine(scale = (0.01, 0.05)))
                    ],

                    random_order = True
                ),
                iaa.Grayscale(alpha = 1)

            ],
            random_order = True

        )
        image_aug,kpsoi_aug = seq(image = image, keypoints = kpsoi)

        return image_aug, kpsoi_aug

    # to run the three functions above
    def __call__(self, output_image_path, binary = 0.7, output_image_size=(2100,2970), start_x = 100, start_y = 160):

        try:
            hkidcard_image_path = self.hkidcard_image_paths[random.randint(0,len(self.hkidcard_image_paths)-1)]
            idcard_image = cv2.imread(hkidcard_image_path)
            # control the size of the signature
            idcard_image = cv2.resize(idcard_image,(random.randint(150,300),random.randint(80,120)))
            # 二值化
            if random.random() < binary:
                idcard_image_binary = cv2.cvtColor(idcard_image, cv2.COLOR_RGB2GRAY)
                ret, idcard_image_binary = cv2.threshold(idcard_image_binary,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if cv2.countNonZero(idcard_image_binary)/(idcard_image_binary.shape[0]*idcard_image_binary.shape[1]) > 0.6:
                    idcard_image = cv2.cvtColor(idcard_image_binary, cv2.COLOR_GRAY2RGB)
                else:
                    idcard_image = idcard_image

            background_img = np.zeros(idcard_image.shape, dtype = np.uint8 ) + random.randint(0,255)
            background_img = cv2.resize(background_img, output_image_size)
           

            height, width, _ = idcard_image.shape
            contour = np.array([(0,0),(width,0),(width,height),(0,height)])

            output_img = image_generator.merge_img(idcard_image, background_img, contour, (start_x,start_y))
            kps = [
                    Keypoint(x = start_x, y = start_y),
                    Keypoint(x = start_x + width/4, y = start_y),
                    Keypoint(x = start_x + width/2, y = start_y),
                    Keypoint(x = start_x + 3*width/4, y = start_y),
                    Keypoint(x = start_x + width, y = start_y),
                    Keypoint(x = start_x + width, y = start_y + height/4),
                    Keypoint(x = start_x + width, y = start_y + height/2),
                    Keypoint(x = start_x + width, y = start_y + 3*height/4),
                    Keypoint(x = start_x + width, y = start_y + height),
                    Keypoint(x = start_x + 3*width/4, y = start_y + height),
                    Keypoint(x = start_x + width/2, y = start_y + height),
                    Keypoint(x = start_x + width/4, y = start_y + height),
                    Keypoint(x = start_x , y = start_y + height),
                    Keypoint(x = start_x , y = start_y + 3*height/4),
                    Keypoint(x = start_x , y = start_y + height/2),
                    Keypoint(x = start_x , y = start_y + height/4),
            
                ]
            image_aug, kpsoi_aug = image_generator.key_aug(output_img,kps)
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2GRAY)
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2BGR)

            image_generator.change_background(image_aug, np.array(kpsoi_aug.to_xy_array().astype(int)),output_image_path=output_image_path)
        except Exception as e:
            print(e)
            self.__call__(output_image_path, output_image_size = (2100,2970),start_x = 1000, start_y = 1600)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "signature generation")
    parser.add_argument("--background_image_path",default = "source\det_background_images\*.jpg", type = str, help = "  ")
    parser.add_argument("--idcard_image_path", default = "debug\idcard_train\*.png", type = str, help = "  ")
    parser.add_argument("--output_image_save_path", default = r"debug\idcard_det_train", type = str, help ="  ")
    parser.add_argument("--images_number", default = 20, type = int, help = "  ")
    parser.add_argument("--multitask", default = 2, type=int, help = "number of tasks")
    args = parser.parse_args() 


    image_generator = DetectImageGenerator(background_image_path_list = glob(args.background_image_path),hkidcard_image_path_list = glob(args.idcard_image_path))

    if not os.path.exists(args.output_image_save_path):
        os.makedirs(args.output_image_save_path)   
        
    def task():
        image_count = int(args.images_number/args.multitask) + 1
        for i in range(image_count):
            print(i)
            width = random.randint(100,220)
            height = int(width*1.414)
            image_name = str(i) + "_" + str(time.time()).replace(".","") + ".jpg"
            save_path = os.path.join(args.output_image_save_path, image_name)
            image_generator(output_image_path = save_path, output_image_size=(width,height), start_x = int(width/3),start_y = int(height/3))

    for num in range(args.multitask):
        t =  threading.Thread(target = task, args=())
        t.start()    
