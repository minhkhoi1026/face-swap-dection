from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

input_file = "data/roopbo/real/050.mp4"
output_file = "data_extract/demo/video.mp4"

start_time = 0  # Start time in seconds
end_time = 1    # End time in seconds

ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
