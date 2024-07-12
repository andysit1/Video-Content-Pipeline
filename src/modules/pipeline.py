"""
BRAINSTORM
  -> goal to structure and refactor the code ina way that allows us to build ontop of what we have without thinking about too much code at once
    -> the code is built based one what I felt like doing and going with the flow but now that we can output/input good results we should clean itup as we
    create harder and more complex features

  -> Base on how we are already processing the code we should break out code into phases ie data process, data gathering, ranking, uploading
      advantage is that we dont have to manually do the annoying setup in main and we can change the structure of our pipeline however we want
      -> I was already thinking smth similar with phases one and two TODO: get rid of phase 1 and 2 folders (redunant)

      -> we can use states we pass the data onto the next and cache the input and output into txt files between stages
      -> if we reuse machine.py and an pipeline_engine to handle events and errors
        -> I think having an engine might be good so we can time (profile code per stage) and im familiar since I used the same for game dev with pygame

  -> Each stage should be using aggregation so we can reduce heaviness of the code and allow for testcases
    -> right off the bat (cache, savable, opencv commands, ffmpeg commands)
      -> NOTE: cache can probably just be a decorator function to push into a file, just name the txt file based on the filename so we dont have fat debugging logs

                              (txt file cache)            (human input)
Type of Stages := Download VOD, Data Cache, Data process, Analyze Data, Action Stage (cutting/trimming), Compile, Upload -> Youtube/Twitch/Medal
                                      (video read clip reading)

"""

#Future -> might need to refactor this code so we better support different functions / use per stage but it's okay for a start
class Stage():
    def __init__(self, engine):
        self.engine = engine

    def on_run(self):
        pass

    def on_error(self):
        #base on the location we can Ok it or return it to the pipeline to decide where to go or change output
        pass


class Machine:
    """
    Manages transitions between different game states.
    """
    def __init__(self):
        """
        Initialize a Machine object.
        """
        self.current : Stage = None
        self.next_state : Stage = None

    def update(self):
        """
        Update the current state.
        """
        if self.next_state:
            self.current = self.next_state
            self.next_state = None


class PipelineEngine:
    def __init__(self):
        self.DEBUG = False
        self.PERFORMANCE = False

    # should logs be stage dependant or pipeline dependent () maybe have both one for stage changes and one for stages functions?
    def save_logs(self):
        pass

    def on_change(self):
        pass


    def loop(self):
        while self.running:
            self.machine.current.run()



    def run(self, state):
        self.machine.current = state
        self.loop()


