import os
from webbrowser import get
import boto3
import datetime
import argparse
from common import is_image
from common import get_py_path
from botocore.exceptions import ClientError
access_key_id = ''
access_pswd = ''
bucket = 'wivisionproduction'

def upload2s3(src_pth, file_name, dest_path):
    res = boto3.resource('s3')
    clt = boto3.client('s3')
    try:
        clt.upload_file(src_pth, bucket, str(dest_path + '/' + file_name))
    except ClientError as e:
        print(e)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='Provide Directory path')
    parser.add_argument('--folder_name', '-fn', help='Folder Name')
    parser.add_argument('--key_id', help='Provide Directory path')
    parser.add_argument('--pswd', help='Provide Directory path')
    args = parser.parse_args()
    src_path = args.src_path
    access_key_id = args.key_id
    access_pswd = args.pswd
    if args.folder_name is None:
        return src_path, None
    else:
        return src_path, args.folder_name

def creat_dt_on_s3():
    res = boto3.resource('s3')
    clt = boto3.client('s3')
    now = datetime.datetime.now()
    dt_str = now.strftime("%d-%m-%y-%H-%M-%S")
    clt.put_object(Bucket = bucket, Key=(dt_str+'/'))
    return dt_str

def create_txt(dir_name):
    with open(get_py_path() + 'latest_dataset.txt', 'a') as f:
        f.writelines('{}'.format(dir_name))
    f.close()
    res = boto3.resource('s3')
    obj = res.Object('wivisionproduction', 'latest_dataset.txt')
    obj.put(Body = open(get_py_path() + 'latest_dataset.txt', 'rb'))

if __name__ == "__main__":
    acc_id = ''
    psw_ = ''
    src_path, dir_ = parse_opt()
    if dir_ is None:
        dir_ = creat_dt_on_s3()
    else:
        pass
    create_txt(dir_)
    if len(os.listdir(src_path)) > 0:
        for file_ in os.listdir(src_path):
            if is_image(file_):
                upload2s3(src_path + file_, file_, dir_)
    