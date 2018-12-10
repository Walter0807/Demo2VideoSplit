# Lecture-Divider👨‍🏫👩‍🏫📹

This is the second part of a final project for the freshmen course "Introduction to Computing"(Honor Track). The project mainly focused on image processing using OpenCV.

Specifically, SpeechDivider suggests the division of a long video(record of a class or presentation) based on the contents on the screen. Ideally, it would divide a presentation into small videos of every slides.

## Implementation

- Extract the screen position as ROI using Canny edge detector
- Then computes the similarity between neighboring ROI
  - Several methods are applied to improve robustness
  - Computing similarity using HSV distance/SIFT/perceptual hashing algorithm
  - Using dHash as similarity function performs best
- Suggest division when the similarity is below threshold

## Sample

A typical frame of the video:

![class.jpg](https://i.loli.net/2018/01/29/5a6e00095e273.jpg)

Extracted ROI

![ROI.jpg](https://i.loli.net/2018/01/29/5a6e0008224ff.jpg)
