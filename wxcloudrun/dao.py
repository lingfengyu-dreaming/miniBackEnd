import logging

from sqlalchemy.exc import OperationalError

from wxcloudrun import db
from wxcloudrun.model import Score

# 初始化日志
logger = logging.getLogger('log')

def insert_score(Score):
    '''
    根据用户id添加记录
    :param:
    :Score:Score对象
    '''
    try:
        db.session.add(Score)
        db.session.commit()
    except OperationalError as e:
        logger.info(f"insert Score error with {e}")

def query_score_by_id(id):
    '''
    根据id查询Score
    :param:
    :id:Score的id
    :return:
    :Score对象
    '''
    try:
        return Score.query.filter(Score.id == id).first()
    except OperationalError as e:
        logger.info(f"query Score by id error with {e}")
        return None

def query_score_by_user(user):
    '''
    根据用户id查询Score
    :param:
    :user:Score的user
    :return:
    :Score数组?
    '''
    try:
        return Score.query.filter(Score.user == user)
    except OperationalError as e:
        logger.info(f"query Score by user openid error with {e}")
        return None

def delete_score_by_id(id):
    '''
    根据id删除记录
    :param:
    :id:Score的id
    '''
    try:
        Score = Score.query.get(id)
        if Score is None:
            return
        db.session.delete(Score)
        db.session.commit()
    except OperationalError as e:
        logger.info(f"delete Score by id error with {e}")

def delete_score_by_user(user):
    '''
    根据user删除记录
    :param:
    :user:Score的user
    '''
    try:
        Score = Score.query.filter(Score.user == user)
        if Score is None:
            return
        db.session.delete(Score)
        db.session.commit()
    except OperationalError as e:
        logger.info(f"delete Score by user error with {e}")

# def query_counterbyid(id):
#     """
#     根据ID查询Counter实体
#     :param id: Counter的ID
#     :return: Counter实体
#     """
#     try:
#         return Counters.query.filter(Counters.id == id).first()
#     except OperationalError as e:
#         logger.info("query_counterbyid errorMsg= {} ".format(e))
#         return None

# def delete_counterbyid(id):
#     """
#     根据ID删除Counter实体
#     :param id: Counter的ID
#     """
#     try:
#         counter = Counters.query.get(id)
#         if counter is None:
#             return
#         db.session.delete(counter)
#         db.session.commit()
#     except OperationalError as e:
#         logger.info("delete_counterbyid errorMsg= {} ".format(e))

# def insert_counter(counter):
#     """
#     插入一个Counter实体
#     :param counter: Counters实体
#     """
#     try:
#         db.session.add(counter)
#         db.session.commit()
#     except OperationalError as e:
#         logger.info("insert_counter errorMsg= {} ".format(e))

# def update_counterbyid(counter):
#     """
#     根据ID更新counter的值
#     :param counter实体
#     """
#     try:
#         counter = query_counterbyid(counter.id)
#         if counter is None:
#             return
#         db.session.flush()
#         db.session.commit()
#     except OperationalError as e:
#         logger.info("update_counterbyid errorMsg= {} ".format(e))
