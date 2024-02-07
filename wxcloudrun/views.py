from datetime import datetime
from flask import render_template, request
from qcloud_cos import CosS3Client, CosClientError, CosServiceError
from run import app
from wxcloudrun.dao import insert_score, query_score_by_id, query_score_by_user, delete_score_by_id, delete_score_by_user
from wxcloudrun.model import Score
from wxcloudrun.response import *
from wxcloudrun.runmodel import test_model
from wxcloudrun.cosbrowser import *
import json

# client = client

# 激活环境
@app.route('/init')
def init():
    '''
    :return: success
    '''
    # 初始化数据库
    # query_score_by_id(1)
    # 下载模型
    # status = download_model
    # if status:
    #     print('下载模型成功')
    #     return make_succ_response({"msg": "初始化成功"})
    # else:
    #     print('下载模型失败')
    #     return make_err_response('初始化失败')
    global client
    client = initcos()
    return make_succ_empty_response()

# 上传图片评分
@app.route('/api/sendImage', methods=['POST'])
def scoreImage():
    """
    :params:
    :input:
    :openid:从用户信息中提取
    :fileid:用户上传图片文件id,从参数中获取
    :return:
    :char:返回字符
    :score:返回成绩
    :time:返回当前时间
    """
    # 获取参数列表
    data = request
    # if data.is_json:
    #     print("原数据为json")
    # else:
    #     print("原数据类型有误")
    # print(data)
    # print("这是打印的get_json的结果", data.get_json())
    params = data.get_json()
    # print(params)
    # 从微信调用
    # try:
    # openid = data.headers['X-WX-OPENID']
    # 从统一小程序调用
    # except KeyError:
    # openid = data.headers['X-WX-UNIONID']
    if 'action' not in params:
        print('缺少action参数')
        return make_err_response('缺少action参数')
    else:
        action = params['action']
    fileid = params['fileid']
    if action == 'score':
        # 下载图片
        status = download_model(client)
        if status:
            status = download_image(client, fileid)
            if status == False:
                print('下载图片失败')
                return make_err_response('服务器下载图片失败')
        else:
            print('下载模型失败')
            return make_err_response('服务器初始化环境失败')
        char, score = test_model()
        # scoreitem = Score()
        # scoreitem.user = openid
        # scoreitem.fileID = fileid
        # scoreitem.char = char
        # scoreitem.score = score
        # insert_score(scoreitem)
    else:
        return make_err_response('action参数错误')
    # time = datetime()
    if char == -1:
        print('识别图片失败')
        return make_err_response('服务器识别图片错误')
    else:
        return score_char_response(char, score)
        # return score_time_response(char, score, time)

# 查询评分
@app.route('/api/checkScore', methods=['POST'])
def queryScore():
    """
    :params:
    :openid:从用户信息中提取
    :id:可选id
    :return:
    :score:评分
    :time:时间
    """
    # 获取参数列表
    data = request
    params = data.get_json()
    # 从微信小程序调用
    try:
        openid = data.headers['X-WX-OPENID']
    # 从统一小程序调用
    except KeyError:
        openid = data.headers['X-WX-UNIONID']
    if 'action' not in params:
        return make_err_response('缺少action参数')
    else:
        action = params['action']
    if action == 'user':
        scoreitem = query_score_by_user(openid)
        if scoreitem is None:
            return make_err_response('未找到数据')
        else:
            return score_time_response(scoreitem.char, scoreitem.score, scoreitem.time)
    elif action == 'id':
        scoreitem = query_score_by_id(params.data['id'])
        if scoreitem is None:
            return make_err_response('未找到数据')
        else:
            return score_time_response(scoreitem.char, scoreitem.score, scoreitem.time)
