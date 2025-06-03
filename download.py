import os
import re
import subprocess
from pytubefix import YouTube
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\n\r\t]', "", name).strip()


def get_all_playlist_video_urls(playlist_url):
    """
    Use yt-dlp to fetch all video URLs from a playlist reliably.
    Requires yt-dlp installed and accessible in PATH.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "-J", "--flat-playlist", playlist_url],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        video_urls = [
            "https://www.youtube.com/watch?v=" + entry["id"]
            for entry in data["entries"]
        ]
        return video_urls
    except Exception as e:
        print(f"Failed to retrieve playlist videos via yt-dlp: {e}")
        return []


def download_and_convert(video_url, output_folder):
    try:
        yt = YouTube(video_url)
        safe_title = sanitize_filename(yt.title)
        output_mp3_path = os.path.join(output_folder, safe_title + ".mp3")

        # Skip if already downloaded
        if os.path.exists(output_mp3_path):
            print(f"Already downloaded, skipping: {safe_title}")
            return yt.title, None

        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if not audio_stream:
            msg = f"No audio stream available for: {yt.title}"
            print(msg)
            return yt.title, msg

        temp_filepath = audio_stream.download(
            output_path=output_folder, filename=safe_title
        )

        # Convert to mp3
        os.system(
            f'ffmpeg -i "{temp_filepath}" -vn -ab 192k -ar 44100 -y "{output_mp3_path}"'
        )
        os.remove(temp_filepath)

        print(f"Downloaded and converted: {yt.title}")
        return yt.title, None
    except Exception as e:
        print(f"Error downloading {video_url}: {e}")
        return video_url, str(e)


def download_youtube_playlist_to_mp3_pytube_parallel(
    playlist_url, output_folder="/home/lpk/music", max_workers=20
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_urls = get_all_playlist_video_urls(playlist_url)
    if not video_urls:
        print("No videos found in the playlist.")
        return

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_and_convert, url, output_folder)
                for url in video_urls
            ]

            for future in as_completed(futures):
                title, error = future.result()
                if error:
                    print(f"Failed: {title} -> {error}")

        print(f"Playlist downloaded to '{output_folder}' successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    playlist_url = input("Enter the YouTube playlist URL: ")
    download_youtube_playlist_to_mp3_pytube_parallel(playlist_url)
