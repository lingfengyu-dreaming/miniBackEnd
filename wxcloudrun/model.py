from datetime import datetime

from wxcloudrun import db

# 计数表
class Counters(db.Model):
    # 设置结构体表格名称
    __tablename__ = 'Counters'

    # 设定结构体对应表格的字段
    id = db.Column(db.Integer, primary_key=True)
    count = db.Column(db.Integer, default=1)
    created_at = db.Column('createdAt', db.TIMESTAMP, nullable=False, default=datetime.now())
    updated_at = db.Column('updatedAt', db.TIMESTAMP, nullable=False, default=datetime.now())

# 用户上传图片评分计分表
class Score(db.Model):
    # 设置表格名称
    __tablename__ = 'ImgScore'
    # 设定字段
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.Text)
    fileID = db.Column(db.Text)
    time = db.Column(db.TIMESTAMP, nullable=True, default=datetime.now())
    char = db.Column(db.Text)
    score = db.Column(db.Integer)
