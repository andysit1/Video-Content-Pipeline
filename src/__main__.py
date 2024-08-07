import logging
from .modules.pipeline_builder import PipelineEngine
from .pipes.dl_stage import DownloadPipe


logging.basicConfig(
    level=logging.INFO,
    filename="./logs.txt",
)

def main():
    # Start the game loop
    #must define output video_name and video location

    engine = PipelineEngine()

    # payload3 = {
    #     "stream_id" : 2209204696,
    #     "is_community" : False,
    #     "video_name" : "cery",
    # }

    # payload4 = {
    #     "stream_id" : 2208310278,
    #     "is_community" : False,
    #     "video_name" : "yvonie",
    # }

    payload5 = {
        "stream_id" : 2209761727,
        "is_community" : False,
        "video_name" : "dyrus",
    }

    logging.info("Starting Program")

    #queue the videos..
    # engine.q.put(payload3)
    # engine.q.put(payload4)
    engine.q.put(payload5)
    engine.run(DownloadPipe(engine=engine))


if __name__ == "__main__":
    main() #./TwitchDownloaderCLI videodownload --id 612942303 -b 0:01:40 -e 0:03:20 -o video.mp4
