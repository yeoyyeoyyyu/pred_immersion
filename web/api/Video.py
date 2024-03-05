from database import LectureMetadata

class VideoManager:
    
    def get_video_path(self, LECTURE_ID):
        try:
            lecture = LectureMetadata.query.filter_by(LECTURE_ID=LECTURE_ID).first()
            if lecture and hasattr(lecture, 'VIDEO_PATH'):
                return lecture.VIDEO_PATH
            return "none"
        except Exception as e:
            print(f"Error retrieving video path: {e}")
            return "none"
