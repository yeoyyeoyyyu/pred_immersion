from flask import request, redirect, url_for
from flask.views import MethodView
from database import db, UserMetaData

class MetadataView(MethodView):
    def post(self):
        USER_NN = request.form['USER_NN']
        BIRTH_DT = request.form['BIRTH_DT']
        PHONE_NUM = request.form['PHONE_NUM']

        new_metadata = UserMetaData(USER_NN=USER_NN, BIRTH_DT=BIRTH_DT, PHONE_NUM=PHONE_NUM)
        db.session.add(new_metadata)
        db.session.commit()

        return redirect(url_for('show_video', filename='1.mp4', metadata_id=new_metadata.METADATA_ID))
