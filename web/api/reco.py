from flask import request, jsonify, render_template, url_for
from flask.views import MethodView
from database import db, LectureMetadata, LOG
from Video import VideoManager

import time

class RecoView(MethodView):
    
    def __init__(self):
        super().__init__()
        self.video_manager = VideoManager()
        
    def post(self):
        data = request.json
        try:
            metadata_id = data.get('metadata_id')
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Invalid metadata_id'})

        video_path = data.get('video_path', '')
        lecture = LectureMetadata.query.filter_by(VIDEO_PATH=video_path).first()
        if not lecture:
            return jsonify({'status': 'error', 'message': 'Lecture not found'})

        immersion_rate = data.get('immersion_rate', 0)

        new_log = LOG(
            METADATA_ID=metadata_id,
            LECTURE_ID=lecture.LECTURE_ID,
            RECO_LECTURE_ID=lecture.LECTURE_ID,
            IMM_RATIO=immersion_rate
        )
        db.session.add(new_log)
        db.session.commit()
        return jsonify({'status': 'success', 'log_id': new_log.LOG_ID})   
    
    def get(self):
        retry_count = 0
        max_retries = 5
        log_id = None
        
        while retry_count < max_retries and log_id is None:
            log_id = request.args.get('log_id')
            retry_count += 1
            time.sleep(1)  # 1초 대기

        if log_id is None:
            return jsonify({'status': 'error', 'message': 'No session ID provided'})

        user_log = LOG.query.get(log_id)
        if not user_log:
            return jsonify({'status': 'error', 'message': 'Session not found'})

        lecture_id = user_log.LECTURE_ID
        video_path = self.video_manager.get_video_path(lecture_id)
        immersion_rate = user_log.IMM_RATIO
        metadata_id = user_log.METADATA_ID
        

        # 추천 비디오 경로 및 다음 단계 URL 설정
        reco_video_path = "url"
        next_video_url = url_for('show_video', filename='2.mp4')
    
        if video_path == 'url':
            next_video_url = url_for('show_video', filename='2.mp4')
            reco_video_path = "url" if immersion_rate >= 0.7 else "url"
        
        elif video_path == 'url':
            next_video_url = url_for('show_video', filename='3.mp4')
            reco_video_path = "url" if immersion_rate >= 0.7 else "url"
        
        elif video_path == 'url':
            next_video_url = "url"
            reco_video_path = "url" if immersion_rate >= 0.7 else "url"
            
        reco_lecture = LectureMetadata.query.filter_by(VIDEO_PATH=reco_video_path).first()
        user_log.RECO_LECTURE_ID = reco_lecture.LECTURE_ID
        db.session.add(user_log)
        db.session.commit()
    
                
        return render_template('reco.html',reco_video_path=reco_video_path, next_video_url=next_video_url,log_id=log_id,metadata_id=metadata_id)
