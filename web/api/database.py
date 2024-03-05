from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class UserMetaData(db.Model):
    METADATA_ID = db.Column(db.Integer, primary_key=True,autoincrement=True)
    USER_NN = db.Column(db.String(20))  
    BIRTH_DT = db.Column(db.String(10))  
    PHONE_NUM = db.Column(db.String(4))
    DEL_YN = db.Column(db.String(1), default='N')
    INSERT_DT = db.Column(db.DateTime, default=datetime.utcnow)

class LectureMetadata(db.Model):
    LECTURE_ID = db.Column(db.Integer,primary_key=True,autoincrement=True)
    VIDEO_PATH = db.Column(db.String(255))
    VIDEO_NAME = db.Column(db.String(255))
    VIDEO_TYPE = db.Column(db.String(20))
    DEL_YN = db.Column(db.String(1), default='N')
    INSERT_DT = db.Column(db.DateTime, default=datetime.utcnow)
    
class LOG(db.Model):
    LOG_ID = db.Column(db.Integer, primary_key=True,autoincrement=True)
    METADATA_ID = db.Column(db.Integer, db.ForeignKey('user_meta_data.METADATA_ID'))
    LECTURE_ID = db.Column(db.Integer, db.ForeignKey('lecture_metadata.LECTURE_ID'))
    RECO_LECTURE_ID = db.Column(db.Integer, db.ForeignKey('lecture_metadata.LECTURE_ID'))
    IMM_RATIO = db.Column(db.Float)
    DEL_YN = db.Column(db.String(1), default='N')
    INSERT_DT = db.Column(db.DateTime, default=datetime.utcnow)

class VideoRating(db.Model):
    RATE_INDEX = db.Column(db.Integer, primary_key=True,autoincrement=True)
    LOG_ID = db.Column(db.Integer, db.ForeignKey('log.LOG_ID')) 
    RATE = db.Column(db.Integer)
    DEL_YN = db.Column(db.String(1), default='N')
    INSERT_DT = db.Column(db.DateTime, default=datetime.utcnow)
        