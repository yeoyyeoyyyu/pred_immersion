from flask import request, jsonify
from flask.views import MethodView
from database import db,VideoRating
from Video import VideoManager

class SaveRatingView(MethodView):
    
    def __init__(self):
        super().__init__()
        self.video_manager = VideoManager()
        
    def post(self):
        data = request.json
        
        log_id = data.get('log_id')
        if not log_id:
            return jsonify({'status': 'error', 'message': 'no data'})


        rating = data.get('rating')

        existing_rating = VideoRating.query.filter_by(LOG_ID=log_id).first()

        if existing_rating:
            # 기존 평가와 새 평가가 다른 경우에만 업데이트
            if existing_rating.RATE != rating:
                existing_rating.RATE = rating
                db.session.commit()
        else:
            # 평가가 존재하지 않으면 새로운 평가 추가
            new_video_rating = VideoRating(LOG_ID=log_id, RATE=rating)
            db.session.add(new_video_rating)
            db.session.commit()
        
        return jsonify({'status': 'success'})
