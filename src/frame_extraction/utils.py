import cv2
import glob


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

  import os

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


def get_files_sorted(path : str):
  import os

  if not os.path.exists(path):
    raise TypeError("Path not found.")





if __name__ == "__main__":
  combine_output_frames_into_video()