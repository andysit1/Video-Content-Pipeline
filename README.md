# Video-Content-Pipeline
Making videos at lightspeed

![Alt text](assets/smth.png)

## Purpose

In this project I wanted to mass edit large videos for streams on twitch for personal and school club reasons. I always thought this project was really interesting which is why I decided to spend some time this summer coding it.

## Highlights

- Takes large videos and filters out silence intervals
- Each interval are consider clips which then is processed futher using opencv image processing algorithms
- Clips then are ranked based on citeria ie loudness, color, and percentage white pixels after thresholding
- Average the points and merge the above average clips together to form a videos

This returns a condense video of key moments within a large stream which can be taken futher for editing or whatever needs.


## Installation

```bash
  pip install -r requirements.txt
```

## Screenshots

![Valorant - Full Screen](assets/processed_valorant_match.PNG)
![Valorant - Crosshair](assets/reyna_orb_proccesed.png)


## Comments
This project is still very new and I still have lots that I want to implement, so come back every week and hopefully it will be nicer. If you have any questions send me an email at andysit173@gmail.com and let me know any feedback or changes you think will help.