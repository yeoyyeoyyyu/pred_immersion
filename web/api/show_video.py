from flask import request, render_template, url_for, abort
from flask.views import MethodView
from database import UserMetaData

class ShowVideoView(MethodView):
    def get(self, filename):
        metadata_id = request.args.get('metadata_id')
        # metadata_id가 데이터베이스에 존재하는지 확인
        metadata = UserMetaData.query.filter_by(METADATA_ID=metadata_id).first()
        
        if not metadata:
            
            return abort(404)  
        video_path = url_for('static', filename='videos/concept/' + filename)
    
        return render_template('new_reco/index.html', video_path=video_path, metadata_id=metadata_id)
