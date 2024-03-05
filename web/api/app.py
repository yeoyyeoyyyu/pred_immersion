from flask import Flask
from database import db
from home import HomeView
from metadata import MetadataView
from show_video import ShowVideoView
from reco import RecoView
from save_rating import SaveRatingView
from send_message import ChatView

app = Flask('app_v6')
app.config['SQLALCHEMY_DATABASE_URI'] = 'DATABASE_ADDRESS'

db.init_app(app)


# URL 라우트와 뷰 클래스 연결
app.add_url_rule('/', view_func=HomeView.as_view('home'))
app.add_url_rule('/metadata', view_func=MetadataView.as_view('metadata'), methods=['POST'])
app.add_url_rule('/<filename>', view_func=ShowVideoView.as_view('show_video'))
app.add_url_rule('/reco', view_func=RecoView.as_view('reco'), methods=['GET', 'POST'])
app.add_url_rule('/save_rating', view_func=SaveRatingView.as_view('save_rating'), methods=['GET','POST'])
app.add_url_rule('/send_message', view_func=ChatView.as_view('send_message'),methods=['POST'])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8333, log_level="debug", reload=True, workers=1)
    

