from datetime import datetime
from flask import render_template, request
from qcloud_cos import CosS3Client, CosClientError, CosServiceError
from run import app
from wxcloudrun.dao import delete_counterbyid, query_counterbyid, insert_counter, update_counterbyid
from wxcloudrun.dao import insert_score, query_score_by_id, query_score_by_user, delete_score_by_id, delete_score_by_user
from wxcloudrun.model import Counters, Score
from wxcloudrun.response import make_succ_empty_response, make_succ_response, make_err_response, score_char_response, score_time_response
from wxcloudrun.runmodel import test_model
from wxcloudrun.cosbrowser import initcos

# 不要动，这里是全局变量
client

# 激活环境
@app.route('/init', methods=['GET'])
def init():
    '''
    :return: success
    '''
    query_score_by_id(1)
    # 初始化cos
    global client
    client = initcos()
    # 下载模型文件
    bucket = "7072-prod-5g5ivxm6945fbe76-1320253797"
    model_path = "model/model-e86.pt"
    local_path = "./model/model.pt"
    try:
        f = open('./model/model.pt', 'r')
        f.close()
    except FileNotFoundError:
        for i in range(0, 3):
            try:
                client.download_file(Bucket=bucket, Key=model_path, DestFilePath=local_path)
                break
            except CosClientError or CosServiceError as e:
                print(e)
        return make_succ_empty_response()

# 上传图片评分
@app.route('/api/sendImage', methods=['POST'])
def scoreImage():
    """
    :params:
    :input:
    :openid:从用户信息中提取
    :fileid:用户上传图片文件id,从参数中获取
    :
    :return:
    :score:返回成绩
    """
    # 从微信调用
    if request.header['X-WX-SOURCE']:
        openid = request.header['X-WX-OPENID']
    # 从统一小程序调用
    else:
        openid = request.header['X-WX-UNIONID']
    params = request.get_json()
    if 'action' not in params:
        return make_err_response('缺少action参数')
    else:
        action = params['action']
    fileid = params['fileid']
    if action == 'score':
        char, score = test_model()
        scoreitem = Score()
        scoreitem.user = openid
        scoreitem.fileid = fileid
        scoreitem.char = char
        scoreitem.score = score
        insert_score(scoreitem)
    else:
        return make_err_response('action参数错误')
    return score_char_response(char, score)

# 查询评分
@app.route('/api/checkScore', method=['POST'])
def queryScore():
    """
    :params:
    :openid:从用户信息中提取
    :id:可选id
    :
    :return:
    :score:评分
    :time:时间
    """
    params = request.get_json()
    # 从微信小程序调用
    if request.header['X-WX-SOURCE']:
        openid = request.header['X-WX-OPENID']
    # 从统一小程序调用
    else:
        openid = request.header['X-WX-UNIONID']
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


@app.route('/')
def index():
    """
    :return: 返回index页面
    """
    return render_template('index.html')


@app.route('/api/count', methods=['POST'])
def count():
    """
    :return:计数结果/清除结果
    """

    # 获取请求体参数
    params = request.get_json()

    # 检查action参数
    if 'action' not in params:
        return make_err_response('缺少action参数')

    # 按照不同的action的值，进行不同的操作
    action = params['action']

    # 执行自增操作
    if action == 'inc':
        counter = query_counterbyid(1)
        if counter is None:
            counter = Counters()
            counter.id = 1
            counter.count = 1
            counter.created_at = datetime.now()
            counter.updated_at = datetime.now()
            insert_counter(counter)
        else:
            counter.id = 1
            counter.count += 1
            counter.updated_at = datetime.now()
            update_counterbyid(counter)
        return make_succ_response(counter.count)

    # 执行清0操作
    elif action == 'clear':
        delete_counterbyid(1)
        return make_succ_empty_response()

    # action参数错误
    else:
        return make_err_response('action参数错误')


@app.route('/api/count', methods=['GET'])
def get_count():
    """
    :return: 计数的值
    """
    counter = Counters.query.filter(Counters.id == 1).first()
    return make_succ_response(0) if counter is None else make_succ_response(counter.count)
