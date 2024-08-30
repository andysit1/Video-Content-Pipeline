import click
from .modules.pipeline_builder import PipelineEngine
import logging
from .pipes.dl_stage import DownloadPipe
from .pipes.entry_stage import EntryPipe
import os
from .constants import OUTPUT_PATH
import questionary

logging.basicConfig(
    level=logging.INFO,
    filename="./logs.txt",
)

def load_config():
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
          return dest_video_output.strip()
        else:
           return line.strip()


@click.command()
@click.option("-c", "--compile", is_flag=True, help="compile clips into video.")
@click.option("-e", "--extract", is_flag=True, help="extract data and analyze for videodrill.")
@click.option("-d", "--dev", is_flag=True, help="dev mode")
def run(compile: bool, extract: bool, dev: bool):
    dest = load_config()

    engine = PipelineEngine()
    engine.output_path = dest
    flags = [compile, extract, dev]
    # Check if exactly one flag is set
    num_flags_set = sum(flags) #will treat false as 0
    if num_flags_set != 1:
        click.echo("Error: Exactly one of flags must be provided.")
        click.get_current_context().exit(1)

    if compile:
       engine.compile = True
       #thus extract must be false
    else:
       engine.compile = False
       #thus extract must be true

    try:
      #change base on your needs when running dev profile
      if dev:
        engine.payload = {
          "is_community" : False,
          'video_name' : "jacob_teach_2",
          "in_filename" : "E:\Projects/2024\wega_transcribe/JacobTeachesVMIX.mkv",
        }

        engine.mode = "extract"
        engine.compile = True
        engine.run(EntryPipe(engine=engine))


      if compile:
        stream_id = questionary.text("Provide Stream ID on Twitch:").ask()
        video_name = questionary.text("Provide name of output video:").ask()

        engine.payload = {
            "stream_id" : stream_id,
            "is_community" : False,
            "video_name" : video_name,
        }

        engine.mode = "compile"

        DownloadPipe(engine=engine)

      if extract:
        in_file = questionary.path("What's the path to the video file?").ask()

        video_name = questionary.text("Provide name of output video:").ask()

        engine.payload = {
          "is_community" : False,
          'video_name' : video_name,
          "in_filename" : in_file,
        }

        engine.mode = "extract"

        engine.run(EntryPipe(engine=engine))

    except Exception as e:
        raise Exception(e)


if __name__ == "__main__":
    run()