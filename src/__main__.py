from .modules.pipeline_builder import PipelineEngine
from .pipes.entry_stage import EntryPipe

def main():
    # Start the game loop

    #must define output video_name and video location

    engine = PipelineEngine()

    in_filename = "E:/Projects/2024/Video-Content-Pipeline/TD/VODS/soda.mp4"
    engine.payload['in_filename'] = in_filename
    engine.payload['video_name'] = 'soda'
    engine.payload['is_community'] = False
    engine.run(EntryPipe(engine=engine))

if __name__ == "__main__":
    main()