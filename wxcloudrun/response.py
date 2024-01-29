import json

from flask import Response


def make_succ_empty_response():
    data = json.dumps({'code': 0, 'data': {}})
    return Response(data, mimetype='application/json')


def make_succ_response(data):
    data = json.dumps({'code': 0, 'data': data})
    return Response(data, mimetype='application/json')


def make_err_response(err_msg):
    data = json.dumps({'code': -1, 'errorMsg': err_msg})
    return Response(data, mimetype='application/json')

def score_char_response(char, score):
    res = {'code': 0, 'char': char, 'score': score}
    data = json.dumps(res)
    return Response(data, mimetype='application/json')

def score_time_response(char, score, time):
    res = {'code': 0, 'char': char, 'score': score, 'time': time}
    data = json.dumps(res)
    return Response(data, mimetype='application/json')
