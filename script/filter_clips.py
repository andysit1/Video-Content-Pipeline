from icecream import ic
import ast
import questionary


def read_lines(path : str) -> list[str]:
    """Read the file and return a list of lines."""
    with open(path, 'r') as file:
        return file.readlines()

import os

#pulled from compile_stage.py
def remove_clip_by_average_point(in_filename : str) -> dict:
  lines = read_lines(in_filename)
  start_size = len(lines)
  data = []

  total_points = 0
  average_points = 0

  for line in lines:
    converted_dict = ast.literal_eval(line) #ast.literal_eval is crazy good
    data.append(converted_dict)
    ic(converted_dict)

    try:
      total_points += converted_dict['points']
    except Exception as e:
       ic(e)
  average_points = total_points / len(lines)

  stats_obj = {
    'total_points' : total_points,
    'average_points' : average_points
  }

  ic("Avg :", stats_obj['average_points'])
  avg = stats_obj['average_points']
  for line in lines:
    converted_dict = ast.literal_eval(line) #ast.literal_eval is crazy good
    if converted_dict['points'] < avg:
      print("Removing converted_dict['name'] since {} is < than {}".format(converted_dict['points'], avg))
      os.remove(converted_dict['name'])

  lines = read_lines(in_filename)
  end_size = len(lines)

  print("Before {}".format(start_size))
  print("End {}".format(end_size))
  print("Culled {}".format(start_size - end_size))



if __name__ == "__main__":
  ic(" Filtering Clips")
  path = r'C:\Users\andys\AppData\Local\vid-flow\private\jacob_teach_2\text_cache\analyze_data.txt'
  stats = remove_clip_by_average_point(path)





