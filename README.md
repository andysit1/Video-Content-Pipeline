# Video-Content-Pipeline
A cli application to create video fast.

Requires ffmpeg and the [TwitchDownloader](https://github.com/lay295/TwitchDownloader) if you plan to pull twitch streams with streamid

## Purpose

In this project I wanted to mass edit large videos for streams on twitch for personal and school club reasons. I always thought this project was really interesting which is why I decided to spend some time this summer working on it.

## How it Works

- Takes large videos and filters out silence intervals
- Each interval are consider clips which then is processed futher using opencv image processing algorithms
- Clips then are ranked based on citeria ie loudness, color, and percentage white pixels after thresholding
- Average the points and merge the above average clips together to form a videos

This returns a condense video of key moments within a large stream which can be taken futher for editing or whatever needs.


## Install

```
pipx vidflow
```

or

for development

```
git clone https://github.com/andysit1/Video-Content-Pipeline.git
pip install -e .
```

At this point, you should have the VidFlow script in the path and you can run VidFlow --help to see the optional flags.

You only need to use VidFlow flags -c or -e which allow you to run the pipeline in two different modes. In compile mode, clips are culled by the average points and combined into a single file. In extract mode, you do everything except compile the video. This mode is used in conjunction with videodrill application inorder to trim the videos quickly. In most cases if you install VidFlow you will be using in -c.

Once, started the program will ask some questions such as twitch streamid, name of video, and it will begin running the script.

___


Features to Add
- compile mode with just a video input file
- add video example in readme.md
