



"""
  Threshold :
  Blurr :

  CLip.set(timestamp)

  Clip.process()
    triggers automates slicing for timestamps
    give videofile new id name and save it in clips

    Now process it in opencv methods
      We want to glur, threshold, color thresholding,

  Clip.get_stats()
    returns the points for overall, specific type of extraction, and more

"""


from frame_extraction.main import ProcessHandler




class Clip:

  def __init__(self):
    self.points : int = None

    #video details
    self.path : str = None
    self.timestamp : str = None

    #opencv specific
    self.processor_handler = ProcessHandler()


  def process(self):
    self.processor_handler.run()
    pass





