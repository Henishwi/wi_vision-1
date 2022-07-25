#Import Libraries
import argparse
import numpy as np
from numpy.linalg import norm
from common import *
import yaml 
from yaml.loader import SafeLoader
from s3_upload import *

#Weight Variables
obj_weight_yolov5 = ''
obj_weight_classifier = ''
pet_weight_classifier = ''
bucketname = 'wivisionproduction'

#DECLARE FUNCTIONS  

def process_src(src_loc):
    try:
        images = []
        if os.path.isdir(src_loc):
            for file_ in os.listdir(src_loc):
                if is_image(file_):
                    images.append(src_loc + file_)
        
        elif is_image(src_loc):
            images.append(src_loc)
        elif is_bag_file(src_loc):
            try:
                o_dir = process_bagfiles(src_loc, '/pylon_camera_node/image_rect_color')
                images = process_src(o_dir)
            except Exception as e:
                print("Bag Images Process error {}".format(e))
        if len(images) > 0:
            return images
        else:
            raise Exception('No image files found in the provided source path.')
    except Exception as e:
        print('process_src function error {}'.format(e))

def process_dest(dest_path):
    dest_loc = dest_path
    try:
        try:
            dt_ = 'OUTPUT'
            if not(os.path.isdir(dest_loc)):
                if dt_ not in os.listdir(get_py_path()):
                    os.mkdir(get_py_path() + dt_)
                    dest_loc = get_py_path() + dt_
                else:
                    dest_loc = get_py_path() + dt_
            else:
                if dt_ not in os.listdir(get_py_path()):
                    os.mkdir(get_py_path() + dt_)
                    dest_loc = get_py_path() + dt_
        except Exception as e:
            print('create default output dir {}'.format(e))

        try:
            if 'crop' not in os.listdir(dest_loc):
                os.mkdir(dest_loc + 'crop/')
                crop_loc = dest_loc + 'crop/'
            else:
                crop_loc = dest_loc + 'crop/'
        except Exception as e:
            print('creating crop {}'.format(e))
        try:
            if 'temp' not in os.listdir(crop_loc):
                os.mkdir(crop_loc + 'temp/')
        except Exception as e:
            print('creating crop {}'.format(e))
        try:
            if 'unknown' not in os.listdir(crop_loc):
                os.mkdir(crop_loc + 'unknown/')
        except Exception as e:
            print('creating crop {}'.format(e))
        try:
            for x in get_labels():
                if x not in os.listdir(crop_loc):
                    os.mkdir(crop_loc + x + '/')
        except Exception as e:
            print('crop label {}'.format(e))
        return dest_loc, crop_loc
    except Exception as e:
        print('process_dest function error {}'.format(e))


#Main Function
def start_detection(img_, dest_loc, crop_loc):
    img_name_ = str(img_.split('/')[-1]).split('.')
    det_before = False
    path_temp = crop_loc + 'temp/'
    detect_object_yolov5(img_, dest_loc)
    if img_name_[0] + '.txt' in os.listdir(dest_loc + 'exp/labels/'):
        det_before = True
        detect_unknown_object(img_, dest_loc, det_before, crop_loc)
    else:
        detect_unknown_object(img_, dest_loc, det_before, crop_loc)
    object_classifier(path_temp, crop_loc, dest_loc)
    if img_name_[0] + '.txt' in os.listdir(dest_loc + 'exp/labels/'):
        img_final = cv2.imread(img_)
        h, w, _ = img_final.shape
        counter = [0,0,0,0,0,0,0,0,0,0,0]
        labels = get_labels()
        label_path = dest_loc + 'exp/labels/' + img_name_[0] + '.txt'
        with open(label_path) as file_:
                lines = file_.readlines()
                img_rect = img_final.copy()
                for line in lines:
                    i, x_cen, y_cen, wi, hi = line.split(' ')
                    i = int(i)
                    x_cen = int(float(x_cen) * w)
                    y_cen = int(float(y_cen) * h)
                    wi = int(float(wi) * w)
                    hi = int(float(hi) * h)
                    #For incrementing counter
                    counter[i] = counter[i] + 1
                    roi = img_final[(y_cen-int(hi/2)+5):(y_cen+int(hi/2)), (x_cen-int(wi/2)+5):(x_cen+int(wi/2))]
                    img_rect = cv2.rectangle(img_rect, [(x_cen-int(wi/2)+5), (y_cen-int(hi/2)+5)], [(x_cen+int(wi/2)),(y_cen+int(hi/2))], (0,255,0), thickness=2)
                    if i != 10:
                        cv2.imwrite(
                            crop_loc + labels[i] + '/' + img_name_[0] + '_' + labels[i]+ '_' + str(counter[i]) + '.jpg', roi)
                    else:
                        cv2.imwrite(
                            crop_loc + 'unknown/' + img_name_[0] + '.jpg', img_rect)
                        copy_to_folder([label_path], crop_loc + 'unknown/')


#Get weight File function
def get_weight_file():
    weights = [get_py_path() + 'wi_required/object_weight_file/best.onnx']
    return weights


#Get Labels
def get_labels():
    labels = []
    with open('obj_yaml.yaml') as f:
        data = yaml.load(f, Loader = SafeLoader)
        labels = data['names']
    return labels

def load_data():
    line = ''
    clt = boto3.client('s3')
    res = boto3.resource('s3')
    bucket = res.Bucket(bucketname) 
    bucket.download_file('latest_dataset.txt', get_py_path() + 'wi_required/latest_dataset.txt')
    with open(get_py_path() + 'wi_required/latest_dataset.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            line = l
    local_dir = None
    for obj in bucket.objects.filter(Prefix=line):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, line))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
    return line

def delete_data():
    pass

#Check All Requirements
def check_requirements():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.system('git clone https://github.com/ultralytics/yolov5')
    req_run = 'pip install -r ' + get_py_path() + 'yolov5/requirements.txt'
    os.system(req_run)
    os.system('git clone https://github.com/Henishwi/wi_required.git')
    """
        -yolov5 cloned and up to date or not
        -wi_required cloned and up to date or not
    """
    pass

#Object Detection Function
def detect_object_yolov5(img_, dest_loc):
    weight = get_weight_file()
    os_run = 'python3 ' + get_py_path() + 'yolov5/detect.py --data ' + get_py_path() + 'wi_required/updated_yaml.yaml --source ' + \
        str(img_) + ' --weights ' + str(weight[0]) + \
        ' --conf 0.25  --save-txt --nosave --project ' + str(dest_loc) + ' --name exp --exist-ok'
    os.system(os_run)

def detect_unknown_object(img_, dest_loc, det_before, crop_loc):
    img_name_ = str(img_.split('/')[-1]).split('.')
    img_ = cv2.imread(img_)
    h, w, _ = img_.shape
    img_ = img_[:, int(w*0.10):int(w-(w*0.10))]
    x = np.average(norm(img_, axis=2)) / np.sqrt(3)

    if x < 70:
        i = 2 * (cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    elif x > 150:
        i = 1/2 * cv2.THRESH_OTSU - cv2.THRESH_BINARY_INV
    else:
        i = cv2.THRESH_BINARY

    img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray_, 110, 255, int(i))
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    h, w, _ = img_.shape
    lst = []
    annot = []
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,wi,hi) = cv2.boundingRect(contour)
        if wi > 100 and hi > 100 and wi < 900 and hi <900:
            lst.append([x,y,x+wi,y+hi])
            annot.append([10, (x+wi/2)/w, (y+hi/2)/h, wi/w, hi/h])
    if det_before:
        og_lst = []
        with open(dest_loc + 'exp/labels/' + img_name_[0] + '.txt') as file_:
            lines = file_.readlines()
            for line in lines:
                temp_ = []
                wi = int(float(line.split(" ")[3]) * w) + 2
                hi = int(float(line.split(" ")[4]) * h) + 2
                x1 = int(float(line.split(" ")[1]) * w) - int(wi/2) + 2
                y1 = int(float(line.split(" ")[2]) * w) - int(hi/2) + 2
                x2 = x1+wi
                y2 = y1+hi
                temp_.append(x1)
                temp_.append(y1)
                temp_.append(x2)
                temp_.append(y2)
                og_lst.append(temp_)
        i = 0
        for t in lst:
            for u in og_lst:
                x0, y0, x1, y1 = t
                x2, y2, x3, y3 = u
                iou = bb_intersection_over_union([x0, y0, x1, x1], [x2, y2, x3, y3])
                if iou *100 > 5:
                    lst.pop(i)
                    annot.pop(i)
                    break
            i += 1
        j = 0
        for ls in annot:
            with open(dest_loc + 'exp/labels/' + img_name_[0] + '.txt', 'a+') as f:
                f.writelines(str(ls[0]) + ' ' + str(ls[1]) + ' ' + str(ls[2]) + ' ' + str(ls[3]) + ' ' + str(ls[4]) + '\n')
            cv2.imwrite(crop_loc + 'temp/' + img_name_[0] + '_' + str(j) + '.jpg', img_[lst[j][0]:lst[j][2],lst[j][1]:lst[j][3]])
            j += 1

    else:
        j = 0
        for ls in annot:
            with open(dest_loc + 'exp/labels/' + img_name_[0] + '.txt', 'a+') as f:
                f.writelines(str(ls[0]) + ' ' + str(ls[1]) + ' ' + str(ls[2]) + ' ' + str(ls[3]) + ' ' + str(ls[4]) + '\n')
            cv2.imwrite(crop_loc + 'temp/' + img_name_[0] + '_' + str(j) + '.jpg', img_[lst[j][0]:lst[j][2],lst[j][1]:lst[j][3]])
            j += 1

def object_classifier(dir_temp, crop_loc, dest_loc):
    pass

#Pet Detection Function
def pet_detection():
    pass

#Color Detection
def color_detection():
    pass

#Argument Parser Function
def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--s3_', default=False ,help="If want to use s3 service.")
    parser.add_argument('--src_loc',  '-sl', help='Provide Source Location of the Image. DIRECTORY; IMAGE FILES OF FORMAT PNG, JPG, JPEG; BAGFILES ARE ACCEPTED.')
    parser.add_argument('--dest_loc', '-dl', default= '', help='Provide DIRECTORY path to save crops. SUGGESTED TO PROVIDE AN EMPTY DIRECTORY')
    parser.add_argument('--pet_det', default=False, help='Mention this to do pet classification')
    arg1 = parser.parse_args()
    if arg1.s3_ == False:
        args = parser.parse_args()
        return args.src_loc, args.dest_loc, args.pet_det, arg1.s3_
    else:
        return arg1.src_loc, None, None, arg1.s3_     



#Main Function
if __name__ == "__main__":
    check_requirements()
    src_loc, dest_loc, pet_det, s3_ = parse_opt()
    if s3_ == False:
        dest_loc, crop_loc = process_dest(dest_path= dest_loc)
        img_arr = process_src(src_loc)
        for img_ in img_arr:
            start_detection(img_, dest_loc, crop_loc)
    else:
        src_loc = get_py_path() + '/' + load_data()
        img_arr = process_src(src_loc)
        dest_loc, crop_loc = process_src(dest_loc)
        for img_ in img_arr:
            start_detection(img_, dest_loc, crop_loc)
        """src_loc, dest_loc, crop_loc = load_data()
        img_arr = process_src(src_loc)
        for img_ in img_arr:
            start_detection(img_, dest_loc, crop_loc)
        pass"""