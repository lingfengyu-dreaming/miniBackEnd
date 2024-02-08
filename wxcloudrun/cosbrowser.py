from qcloud_cos import CosConfig, CosS3Client, CosClientError, CosServiceError
import sys
import os
import logging
import requests
import json

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
# logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger('log')

# global client

# 初始化cos
def initcos():
    url = "http://api.weixin.qq.com/_/cos/getauth"
    res = requests.get(url)
    info = res.json()
    sec_id = info['TmpSecretId']
    sec_key = info['TmpSecretKey']
    token = info['Token']
    time = info['ExpiredTime']
    region = 'ap-shanghai'
    config = CosConfig(Region=region, SecretId=sec_id, SecretKey=sec_key, Token=token, Timeout=time)
    # global client
    client = CosS3Client(config)
    return client

# 下载模型
def download_model(client):
    # 初始化cos
    # client = initcos()
    # 下载模型文件
    bucket = '7072-prod-5g5ivxm6945fbe76-1320253797'
    model_path = 'model/model-e86.pt'
    local_path = 'model/model.pt'
    status = os.path.exists('model/model.pt')
    if status:
        print("model load true")
    else:
        for i in range(10):
            try:
                client.download_file(Bucket=bucket, Key=model_path, DestFilePath=local_path)
                # res = client.get_object(Bucket=bucket, Key=model_path)
                # res['Body'].get_stream_to_file(local_path)
                log.info("model download true")
                print('model download true')
                return True
            except CosClientError or CosServiceError as e:
                log.error(e)
                print(e)
                continue
    return False

# 下载图片
def download_image(client, fileid):
    ls = fileid.split('/')
    bucket = ls[2].split('.')[1]
    file_path = ls[3] + '/' + ls[4] + '/' + ls[5] + '/' + ls[6] + '/' + ls[7]
    local_path = "image/img.jpg"
    # client = initcos()
    for i in range(10):
        try:
            client.download_file(Bucket=bucket, Key=file_path, DestFilePath=local_path)
            # res = client.get_object(Bucket=bucket, Key=file_path)
            # res['Body'].get_stream_to_file(local_path)
            # log.info("image download true")
            # print(f'image download true')
            return True
        except CosClientError or CosServiceError as e:
            log.error(e)
            print(e)
            continue
    return False
