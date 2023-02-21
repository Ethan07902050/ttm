import os, sys
from moviepy.editor import VideoFileClip
###
# video_file_dir = "/home/kszuyen/DLCV/final/student_data/student_data/videos"
# audio_file_dir = "/home/kszuyen/DLCV/final/student_data/student_data/audios"

video_file_dir = sys.argv[1]
output_root_dir = sys.argv[2]
audio_file_dir = os.path.join(output_root_dir, "dlcvchallenge1_audios")
###

if not os.path.isdir(audio_file_dir):
    os.makedirs(audio_file_dir)

for video_name in os.listdir(video_file_dir):
    video_hashcode = video_name.split('.')[0]
    video = VideoFileClip(os.path.join(video_file_dir, video_name))
    audio = video.audio

    audio.write_audiofile(os.path.join(audio_file_dir, video_hashcode+".wav"))



