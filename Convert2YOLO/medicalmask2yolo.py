from xml.etree import ElementTree
import os
import numpy as np
import cv2
import argparse

from pathlib import Path

class KaggleMedicalMaskReader():
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
    def get_image_metafile(self, image_file):
        image_name = os.path.splitext(image_file)[0]
        return os.path.join(self.labels_dir, str(image_name+'.xml'))
    
    def get_data_attributes(self):
        image_extensions = ['.jpeg', '.jpg', '.png']
        _count_mask = 0
        _count_no_mask = 0
        for image_name in os.listdir(self.images_dir):
            if image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_full_path = os.path.join(self.images_dir, image_name)
                label_full_path = Path(image_full_path).with_suffix('.txt')
                
                img = cv2.imread(image_full_path)
                
                if img is None:
                    print(image_full_path)
                    continue
                
                img_h, img_w = img.shape[:2]
                
                labels_xml = self.get_image_metafile(image_file=image_name)
                
                if not os.path.isfile(labels_xml):
                    continue
                
                labels = ElementTree.parse(labels_xml).getroot()
                
                with open(label_full_path, 'w') as f:
                    for object_tag in labels.findall("object"):
                        category_name = object_tag.find("name").text

                        if category_name == 'mask':
                            yolo_class = 1
                            color = (0, 0, 255)
                        elif category_name == 'none':
                            yolo_class = 0
                            color = (0, 255, 0)
                        else:
                            continue
                        
                        xmin = int(object_tag.find("bndbox/xmin").text)
                        xmax = int(object_tag.find("bndbox/xmax").text)
                        ymin = int(object_tag.find("bndbox/ymin").text)
                        ymax = int(object_tag.find("bndbox/ymax").text)
                        bbox = [xmin, ymin, xmax, ymax]

                        cx = (xmin+xmax)/2/img_w
                        cy = (ymin+ymax)/2/img_h
                        w = (xmax-xmin)/img_w
                        h = (ymax-ymin)/img_h
                    
                        f.write(f"{yolo_class} {cx} {cy} {w} {h}\n")
                    
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=3)

                cv2.imshow("img", img)
                cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description='Kaggle Medical Mask dataset annotation to YOLO')
    parser.add_argument('--base-dir', default="./MedicalMaskDataset-kaggle/medical-masks-dataset/", help='Location of dataset directory')
    opt = parser.parse_args()
    
    images_dir = os.path.join(opt.base_dir, 'images')
    labels_dir = os.path.join(opt.base_dir, 'labels')
    
    reader = KaggleMedicalMaskReader(images_dir, labels_dir)
    reader.get_data_attributes()

if __name__ == '__main__':
    main()
