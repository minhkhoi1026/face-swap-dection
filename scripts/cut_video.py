import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cut_video(input_file, output_file, start_time, end_time):
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

# Thư mục chứa các video cần cắt
input_folder = 'data/roop/real'
output_folder = 'data/roop_cut/real'

# Thời gian bắt đầu và kết thúc đoạn video cần cắt
start_time = 0  # Thời gian bắt đầu (giây)
end_time = 10   # Thời gian kết thúc (giây)

# Duyệt qua tất cả các tệp tin trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):  # Chỉ xử lý các tệp tin video có định dạng mp4
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        
        # Gọi hàm cắt video
        cut_video(input_file, output_file, start_time, end_time)
