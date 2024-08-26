import click
from .modules.pipeline_builder import PipelineEngine
import logging
from .pipes.dl_stage import DownloadPipe
from .pipes.entry_stage import EntryPipe

import questionary

logging.basicConfig(
    level=logging.INFO,
    filename="./logs.txt",
)



@click.command()
@click.option("-c", "--compile", is_flag=True, help="compile clips into video.")
@click.option("-e", "--extract", is_flag=True, help="extract data and analyze for videodrill.")
@click.option("-d", "--dev", is_flag=True, help="dev mode")
def run(compile: bool, extract: bool, dev: bool):

    engine = PipelineEngine()

    """
      for now do not give all options for sake of development
    """
    try:

      if dev:
        engine.payload = {
          "is_community" : False,
          'video_name' : "qttest",
          "in_filename" : "./TD/imaqtpie2.mp4",
        }

        engine.run(EntryPipe(engine=engine))

      if compile:
        stream_id = questionary.text("Provide Stream ID on Twitch:").ask()
        video_name = questionary.text("Provide name of output video:").ask()

        engine.payload = {
            "stream_id" : stream_id,
            "is_community" : False,
            "video_name" : video_name,
        }

        DownloadPipe(engine=engine)

      if extract:
        in_file = questionary.path("What's the path to the video file?").ask()

        video_name = questionary.text("Provide name of output video:").ask()

        engine.payload = {
          "is_community" : False,
          'video_name' : video_name,
          "in_filename" : in_file,
        }

        engine.run(EntryPipe(engine=engine))

    except Exception as e:
        raise Exception(e)


if __name__ == "__main__":
    run()