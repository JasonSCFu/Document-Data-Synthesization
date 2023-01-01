import os
from glob import glob
import json
import numpy as np

class Convert2COCO():
    def __init__(self,image_path,annot_path,json_path = "source\person_keypoints.json"):
        self.annot_path = annot_path
        self.coco_annots = {}
        self.coco_annots["info"] = {"description": "",
                                    "url": "",
                                    "version": "1.0",
                                    "year":2022,
                                    "date_created":"2022/06/02",
                                    "contributor":""
                                    }
        self.coco_annots["licenses"] = [{"url": "",
                                        "id":1,
                                        "name":""}]
        self.coco_annots["categories"] = [{"supercategory":"default",
                                            "id":1,
                                            ## I change to Signature, need to consistent with model config file
                                            "name":"Signature",
                                            "keypoints":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"],\
                                            "skeleton":[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],\
                                                [10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,1]]  
                                                }]

        self.coco_annots['images'] = []
        self.coco_annots['annotations'] = []
        self.image_path = image_path
        self.num_joints = 16


    def get_annot_from_file(self, annot_path):
        anno = []
        with open(annot_path, "r", encoding = "utf-8") as f:
            label = json.load(f)
            file_name = label["file_name"].replace(".jpg","")
            anno.append([file_name,label['width'], label['height']])
            keypoints = list(label['keypoints'][0])
            kp = [file_name]
            for i in range(16):
                for j in range(2):
                    kp.append(keypoints[i][j])
            anno.append(kp)
            box = label['box2'][0]
            anno.append([file_name, box[0][0],box[0][1],box[1][0],box[1][1]])

        return anno

    def get_annots_from_json(self):
        annot_paths = glob(os.path.join(self.annot_path, "*.json"))
        annots = []
        for annot_path in annot_paths:
            annot = self.get_annot_from_file(annot_path)
            annots.append(annot)
        return annots

    def keep_in(self, points, boxes):
        if len(boxes) == 1:
            box_keep = boxes
        else:
            points = np.array(points).reshape(-1,2)
            px1, py1 = np.amin(points, axis = 0)
            px2, py2 = np.amax(points, axis = 0)
            boxes = np.array(boxes)
            x1,y1,x2,y2 = boxes[:,0]-1, boxes[:,1]-1,boxes[:,2]+1, boxes[:,3]+1
            logical_and = (x1 <= px1) * (y1 <= py1) * (x2 >= px2) * (y2 >= py2)
            box_keep = boxes[logical_and]
        if len(box_keep) == 0:
            return None
        else:
            return box_keep[0]

    def to_vis_points(self,points):
        points = np.array(points, dtype = np.float32).reshape(-1,2)
        points = np.concatenate((points,np.ones((len(points),1),dtype = np.float32)), axis = 1)
        points = points.flatten().tolist()
        return points

    def cvt_one_sample(self, annot, image_id, annot_id):
        image_name = annot[0][0] + ".jpg" if isinstance(annot[0], list) else annot[0] + ".jpg"
        _, img_h, img_w = annot[0]
        coco_image = {"license":1,
                      #"file_name": os.path.join(self.image_path, image_name.split("\\")[-1]),
                      # need to replace the image_path info
                      "file_name": image_name.split("\\")[-1],
                      "coco_url": "",
                      "height": img_h,
                      "width" : img_w,
                      "date_captured": "",
                      "flickr_url" : "",
                      "id": image_id
                        }
        annot_list = []

        keyps = []
        boxes = []
        for i, item in enumerate(annot):
            if len(item) == 2*self.num_joints + 1:
                keyps.append(item[1:])
            elif len(item) == 5:
                boxes.append(item[1:])

        if len(boxes) > 0:
            for points in keyps:
                box_keep = self.keep_in(points, boxes)
                if box_keep is None:
                    continue
                x1,y1,x2,y2 = box_keep
                pseudo_seg = [x1,y1,x2,y1,x2,y2,x1,y2]
                points = self.to_vis_points(points)

                coco_annot = {"segmentation": [pseudo_seg],
                                  "num_keypoints" : self.num_joints,
                                  "area": (x2-x1)*(y2-y1),
                                  "iscrowd" : 0,
                                  "keypoints" : points,
                                  "image_id": image_id,
                                  "bbox" : [x1,y1,x2-x1,y2-y1],
                                  "category_id":1,
                                  "id":annot_id
                                }     
                annot_list.append(coco_annot)
                annot_id += 1
        return coco_image, annot_list, annot_id

    def generat_coco(self, save_coco_path = "train2017.json",print_iter = 1,image_id = 0, annot_id = 0):
        coco_annots = self.coco_annots
        annots = self.get_annots_from_json()
        total_samples = len(annots)

        if total_samples == 0:
            print("total_samples = 0 !! check anno path")
            return
        for i, annot in enumerate(annots):
            if print_iter != -1 and (i+1) % print_iter == 0:
                print("{}/{}".format(i+1, total_samples))
            coco_image, annot_list, annot_id = self.cvt_one_sample(annot, image_id, annot_id)

            if coco_image ==[]:
                print("images error")
            else:
                coco_annots["images"].append(coco_image)

            if annot_list == []:
                print("images error")
            else:
                coco_annots["annotations"].extend(annot_list)
            image_id += 1 
        with open(save_coco_path, "w") as f:
            json.dump(coco_annots,f)

if __name__ == "__main__":
    cvt = Convert2COCO(image_path = "debug\idcard_det_train",\
                       annot_path = "debug\idcard_det_train")
    coco_save_path = r"debug\\card_coco\\test"
    if not os.path.exists(coco_save_path):
        os.makedirs(coco_save_path)
    print(os.path.join(coco_save_path,"train2017.json"))
    cvt.generat_coco(save_coco_path=coco_save_path + "\\train2017.json")