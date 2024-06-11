
#input

#PARSING
#average db in video inorder to pickout ideal threshholds and durations
#implent a freeze detections
  #good way to reduce videos/filter out bad clips with freezedetect ffmpeg

#FOCUS
#implement background subtraction functions with opencv
  #this allows us to compare changes from before and after to see the biggest delta
  #allowing us to pick out the more exciting/ACTION PACK moments of a clip...

#output
  #glob all files in dir into video
  #can probably just find a ffmpeg command to do this...

from base.state_machine import Engine, State


engine = Engine()

class ExtractState(State):
  def __init__(self, engine):
    super().__init__(engine)
    self.tag : str = "end"

  def on_update(self):
    print("Thank you for running our program; Ending.")


class EndState(State):
  def __init__(self, engine):
    super().__init__(engine)
    self.tag : str = "end"

  def on_update(self):
    print("Thank you for running our program; Ending.")




end = EndState(engine=engine)

def main():
  engine.run(end)

if __name__ == "__main__":
  main()