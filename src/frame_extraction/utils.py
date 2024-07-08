import cv2
import glob
import os
from icecream import ic
from phase_one_extract.utils import _logged_popen
import ffmpeg
#rbg to gray scale

def average_method(x_list):
  return sum(x_list) / len(x_list)

def luminosity_method(x_list):
  if len(x_list) != 3:
    raise ValueError("List type is wrong, not enough values (3) found {}".format(len(x_list)))

  return 0.3 * x_list[0] + 0.59 * x_list[1] + 0.11 * x_list[2]


#given a file with pngs we can take the frames and create images
# this will always used out_frames and shot to out_video_path

def combine_output_frames_into_video(frame_rate : int = 60):

  input_video_path = './frame_extraction/out_frame/'


  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  output_video_path = '../processed-output-videos'

  if not os.path.exists(input_video_path):
    raise TypeError("Input Frame Path does not exists")

  if not os.path.exists(output_video_path):
    raise TypeError("Output Frame Path does not exists")

  frames = sorted(glob.glob(os.path.join(input_video_path, "*png")), key=os.path.getmtime)
  print(frames)
  next_index_for_processed_videos = len(os.listdir(output_video_path))
  frame = cv2.imread(frames[0])
  height, width, _ = frame.shape

  output_video_path = '../processed-output-videos/test{}.mp4'.format(next_index_for_processed_videos)
  video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

  for frame in frames:
    frame = cv2.imread(frame)
    video.write(frame)

  video.release()



def fix_window_path(path: str):
  return path.replace("\\", "/")


#ref: https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
def concat_demuxer_method(clips : list):
  # for clip in clips:
  with open('tmp_file.txt', 'w+') as f:
    for clip in clips:
      path = fix_window_path(clip[0])
      ic(path)

      f.write("file {}\n".format(os.path.join(path)))

  f.close()

  #ffmpeg -f concat -safe 0 -i tmp_file.txt -c copy output.mp4
  try:
    _logged_popen(
      (
        ffmpeg
          .input("tmp_file.txt", safe=0, f="concat")
          .output("test.mp4", vcodec="copy")
          .overwrite_output()
          .compile()
      )
    ).communicate()

  except Exception as e:
    print("Error", e)


#pass the parse contents from ranker.get_points()
def combine_select_output_videos_into_video(clips, frame_rate : int = 60):
  ic()

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  output_video_path = '../processed-output-videos'


  if not os.path.exists(output_video_path):
    raise TypeError("Input Frame Path does not exists")

  if not os.path.exists(output_video_path):
    raise TypeError("Output Frame Path does not exists")

  next_index_for_processed_videos = len(os.listdir(output_video_path))
  output_video_path = '../processed-output-videos/final{}.mp4'.format(next_index_for_processed_videos)


  width, height = 1920, 1080
  video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

  #for each clip, get the video + read each frame THEN write into video
  for clip in clips:
    cap = cv2.VideoCapture(clip[0])
    while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        video.write(frame)
      else:
        print("Video ended break...")
        cap.release()

  video.release()

def get_files_sorted(path : str):
  import os

  if not os.path.exists(path):
    raise TypeError("Path not found.")

if __name__ == "__main__":
  combine_output_frames_into_video()