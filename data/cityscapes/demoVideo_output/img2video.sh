#!/usr/bin/env bash

ffmpeg -framerate 17 -pattern_type glob -i '*.png' -c:v libx264 out.mp4
ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -i out.mp4 -vf 'scale=iw*min(1280/iw\,960/ih):ih*min(1280/iw\,960/ih),pad=1280:960:(1280-iw)/2:(960-ih)/2,fps=30' -pix_fmt yuv420p -c:v nvenc_h264 -c:a aac -crf 30 ../demoVideo_output.mp4

