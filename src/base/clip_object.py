



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


class Clip:

  def __init__(self, path : str):
    self.path : str = path

    self.points : int = None
    #video details
    self.timestamp : str = None



  def get_path(self):
    return self.path

  def process(self):
    # self.processor_handler.run()
    pass





