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

"""