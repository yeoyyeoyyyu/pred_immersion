from flask import request, jsonify
from flask.views import MethodView
from google.cloud import dialogflow
import os

# Dialogflow 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_api_설정파일_경로'
project_id = 'project'
language_code = 'ko'

class ChatView(MethodView):

    def post(self):
        data = request.json
        message = data.get('message')
        # 클라이언트로부터 전달받은 metadata_id를 사용하여 session_id 생성
        metadata_id = data.get('metadata_id', 'default_session')  # 기본값 설정

        print(f'Received message: {message} for session {metadata_id}')

        # Dialogflow로부터 응답 받기
        response_message = self.detect_intent_from_text(message, metadata_id)

        return jsonify({'status': 'success', 'message': response_message})

    def detect_intent_from_text(self, text, session_id):
        """Dialogflow의 detect intent 메서드를 호출"""
        
        # Dialogflow 세션 클라이언트 초기화
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        # Dialogflow에 전송할 텍스트 입력
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)

        return response.query_result.fulfillment_text