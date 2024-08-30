from pathlib import Path
from platformdirs import user_config_dir


CONFIG_PATH = Path(
    user_config_dir(appname="vid-flow", appauthor=False, ensure_exists=True)
)

COMMUNITY = "community"
PRIVATE = "private"
OUTPUT = "output-location.txt"


COMMUNITY_PATH = CONFIG_PATH / COMMUNITY
PRIVATE_PATH = CONFIG_PATH / PRIVATE

OUTPUT_PATH = CONFIG_PATH / OUTPUT


if __name__ == "__main__":
  print(CONFIG_PATH)
  print(OUTPUT_PATH)

  import questionary
  def load_config():
    import os
    if not os.path.exists(OUTPUT_PATH):
      with open(OUTPUT_PATH, "w") as f:
         f.write("empty")
         f.close()

    with open(OUTPUT_PATH, "r+") as f:
        line = f.readline()
        if line == "empty":
          print("There is not output dir for clips and videos yet!")
          dest_video_output = questionary.path("Please select the path you want").ask()
          f.truncate(0)
          f.write(dest_video_output)
        else:
           return line
