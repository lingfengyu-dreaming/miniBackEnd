from qcloud_cos import CosConfig, CosS3Client
import sys
import os
import logging
import requests
import json

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
# logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger('log')

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
    client = CosS3Client(config)
    return client
