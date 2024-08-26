import logging
from .modules.pipeline_builder import PipelineEngine
from .pipes.dl_stage import DownloadPipe


logging.basicConfig(
    level=logging.INFO,
    filename="./logs.txt",
)


from .cli import run

def main():
    run()

if __name__ == "__main__":
    main() #./TwitchDownloaderCLI videodownload --id 612942303 -b 0:01:40 -e 0:03:20 -o video.mp4
