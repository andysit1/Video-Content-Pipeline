from .modules.pipeline_builder import PipelineEngine
from .stages.test_stage import FillerPipe

def main():
    # Start the game loop
    engine = PipelineEngine()
    pipe1 = FillerPipe(engine)
    engine.run(pipe1)

if __name__ == "__main__":
    main()