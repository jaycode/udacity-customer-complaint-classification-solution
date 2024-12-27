import os
from moviepy import VideoFileClip

def convert_mp4_to_mp3_in_directory(directory):
    """
    Converts all MP4 files in the given directory to MP3 format.

    Parameters:
        directory (str): Path to the directory containing MP4 files.
    """
    if not os.path.isdir(directory):
        print(f"The provided path '{directory}' is not a valid directory.")
        return

    # Get a list of all MP4 files in the directory
    mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

    if not mp4_files:
        print("No MP4 files found in the directory.")
        return

    print(f"Found {len(mp4_files)} MP4 file(s). Converting to MP3...")

    for mp4_file in mp4_files:
        mp4_path = os.path.join(directory, mp4_file)
        mp3_file = os.path.splitext(mp4_file)[0] + '.mp3'
        mp3_path = os.path.join(directory, mp3_file)

        try:
            # Load the video file and extract audio
            video = VideoFileClip(mp4_path)
            video.audio.write_audiofile(mp3_path)
            print(f"Converted: {mp4_file} -> {mp3_file}")
        except Exception as e:
            print(f"Failed to convert {mp4_file}: {e}")
        finally:
            video.close()

directory_path = "audio" 
convert_mp4_to_mp3_in_directory(directory_path)
