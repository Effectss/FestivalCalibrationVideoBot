import discord
import os
import logging
import cv2
import numpy as np
from discord import app_commands
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
from scipy.signal import spectrogram, medfilt, find_peaks
from pydub import AudioSegment
from PIL import Image
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load token from environment variable
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Ensure output directory exists
output_dir = 'output_files'
os.makedirs(output_dir, exist_ok=True)

class MyBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

bot = MyBot()

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user}')

@bot.event
async def on_guild_join(guild: discord.Guild):
    await bot.tree.sync(guild=guild)

@bot.tree.command(name="process_video", description="Process the uploaded video to find beep after rectangle or use mobile detection")
async def process_video(interaction: discord.Interaction, video: discord.Attachment, mobile: bool):
    try:
        await interaction.response.defer()

        # Download the video
        video_bytes = await video.read()
        video_path = os.path.join(output_dir, 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        # Extract audio and analyze for beep
        audio_path = os.path.join(output_dir, 'audio.wav')
        extract_audio(video_path, audio_path)
        
        # Analyze video for gradient rectangle based on mobile flag
        if mobile:
            rectangle_time, rectangle_image = find_mobile_gradient_rectangle_time(video_path)
        else:
            rectangle_time, rectangle_image = find_gradient_rectangle_time(video_path)

        if rectangle_time is None:
            await interaction.followup.send("Failed to detect the gradient rectangle in the video.")
            return

        # Detect beep closest to the rectangle
        time_difference, spectrogram_path, peaks_plot_path = find_beep_time_after(rectangle_time, audio_path)

        if time_difference is not None:
            await interaction.followup.send(f"Time difference between closest beep and rectangle: {time_difference} ms")

            # Save and send the rectangle image
            if rectangle_image:
                image_path = os.path.join(output_dir, 'detected_rectangle.png')
                rectangle_image.save(image_path)
                with open(image_path, 'rb') as f:
                    await interaction.followup.send(file=discord.File(f, filename='detected_rectangle.png'))
            
            # Send the spectrogram and peaks plot images
            with open(spectrogram_path, 'rb') as f:
                await interaction.followup.send(file=discord.File(f, filename='spectrogram.png'))

            with open(peaks_plot_path, 'rb') as f:
                await interaction.followup.send(file=discord.File(f, filename='peaks_plot.png'))

            # Extract and send the beep audio snippet
            beep_audio_path = os.path.join(output_dir, 'beep_audio.wav')
            extract_beep_audio(audio_path, rectangle_time + time_difference, beep_audio_path)
            with open(beep_audio_path, 'rb') as f:
                await interaction.followup.send(file=discord.File(f, filename='beep_audio.wav'))
        else:
            await interaction.followup.send("Failed to detect a beep after the gradient rectangle.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        await interaction.followup.send(f"An error occurred: {str(e)}")

def extract_audio(input_video_path, output_audio_path):
    """Extract audio from the video and save it to a file."""
    try:
        clip = VideoFileClip(input_video_path)
        clip.audio.write_audiofile(output_audio_path)
        logging.info(f"Extracted audio saved at {output_audio_path}")
    except Exception as e:
        logging.error(f"Failed to extract audio: {str(e)}")
        raise

def find_beep_time_after(rectangle_time, audio_path, target_freq=14300, freq_range=200, dB_threshold=0):
    """Find the time of the beep closest to the rectangle time."""
    try:
        sample_rate, data = wavfile.read(audio_path)
        
        # Ensure data is 1-D
        if len(data.shape) > 1:
            data = data[:, 0]

        # Compute the spectrogram
        frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=1024, noverlap=512)
        
        # Save the spectrogram image
        spectrogram_path = os.path.join(output_dir, 'spectrogram.png')
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.savefig(spectrogram_path)
        plt.close()

        # Convert dB threshold to linear scale
        amplitude_threshold = 10 ** (dB_threshold / 20.0)

        # Define the frequency range for detection
        freq_min = target_freq - freq_range
        freq_max = target_freq + freq_range

        # Find indices of the frequencies within the specified range
        freq_idx_range = np.where((frequencies >= freq_min) & (frequencies <= freq_max))

        if len(freq_idx_range[0]) == 0:
            logging.warning("No frequencies in the specified range.")
            return None, spectrogram_path, None

        # Extract the amplitude spectrum for the target frequency range
        target_amplitude_range = np.max(Sxx[freq_idx_range], axis=0)
        
        # Apply amplitude threshold to filter out low amplitude values
        target_amplitude_range[target_amplitude_range < amplitude_threshold] = 0

        # Apply median filtering for smoothing
        smooth_amplitude = medfilt(target_amplitude_range, kernel_size=9)
        
        # Find peaks in the smoothed amplitude spectrum
        peak_indices, _ = find_peaks(smooth_amplitude, height=amplitude_threshold)

        # Plot the peaks and save the image
        peaks_plot_path = os.path.join(output_dir, 'peaks_plot.png')
        plt.figure(figsize=(10, 4))
        plt.plot(times, smooth_amplitude, label='Smoothed Amplitude')
        plt.plot(times[peak_indices], smooth_amplitude[peak_indices], 'r.', markersize=10, label='Detected Beeps')
        plt.axvline(x=rectangle_time / 1000, color='g', linestyle='--', label='Rectangle Detected')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Beep Detection')
        plt.savefig(peaks_plot_path)
        plt.close()

        # Filter peaks to find beeps after the rectangle time
        beep_times = [times[idx] * 1000 for idx in peak_indices if times[idx] * 1000 > rectangle_time]
        
        if beep_times:
            # Find the beep closest to the rectangle
            closest_beep_time = min(beep_times, key=lambda t: abs(t - rectangle_time))
            # Calculate the time difference in milliseconds
            time_difference = closest_beep_time - rectangle_time
            logging.info(f"Time difference between closest beep and rectangle: {time_difference} ms")
            return time_difference, spectrogram_path, peaks_plot_path

        logging.info("No beeps detected after rectangle")
        return None, spectrogram_path, peaks_plot_path
    except Exception as e:
        logging.error(f"Failed to find beep time: {str(e)}")
        raise

def find_gradient_rectangle_time(video_path):
    """Find the time of the gradient rectangle in the video."""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Load the key image and preprocess it
        key_image_path = r'insert file path here'
        key_image = cv2.imread(key_image_path, cv2.IMREAD_GRAYSCALE)
        key_image = cv2.GaussianBlur(key_image, (5, 5), 0)

        if key_image is None:
            logging.error(f"Error: Unable to load the key image from path {key_image_path}")
            return None, None

        key_height, key_width = key_image.shape

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        rectangle_time = None
        detected_image = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Sample every 3rd frame for better performance
            if frame_count % 3 != 0:
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

            # Template matching
            result = cv2.matchTemplate(frame_gray, key_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= 0.85:  # Increased threshold for better accuracy
                top_left = max_loc
                bottom_right = (top_left[0] + key_width, top_left[1] + key_height)

                detected_image = cv2.rectangle(frame.copy(), top_left, bottom_right, (0, 255, 0), 2)
                detected_image = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
                
                # Calculate time
                rectangle_time = (frame_count / frame_rate) * 1000
                logging.info(f"Rectangle detected at {rectangle_time} ms")
                break

        cap.release()

        if rectangle_time is None:
            logging.info("No rectangle detected.")
            return None, None

        return rectangle_time, detected_image
    except Exception as e:
        logging.error(f"Failed to find gradient rectangle: {str(e)}")
        raise

def find_mobile_gradient_rectangle_time(video_path):
    """Mobile-specific detection of the gradient rectangle in the video."""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Load the key image and preprocess it for mobile (lower resolution, different dimensions, etc.)
        key_image_path = r'insert file path here'  # Mobile-specific key image
        key_image = cv2.imread(key_image_path, cv2.IMREAD_GRAYSCALE)
        key_image = cv2.GaussianBlur(key_image, (3, 3), 0)

        if key_image is None:
            logging.error(f"Error: Unable to load the key image from path {key_image_path}")
            return None, None

        key_height, key_width = key_image.shape

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        rectangle_time = None
        detected_image = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Sample every 2nd frame for better performance on mobile videos
            if frame_count % 2 != 0:
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)  # Adjust GaussianBlur for mobile resolution

            # Template matching
            result = cv2.matchTemplate(frame_gray, key_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= 0.80:  # Lower threshold for mobile-specific videos
                top_left = max_loc
                bottom_right = (top_left[0] + key_width, top_left[1] + key_height)

                detected_image = cv2.rectangle(frame.copy(), top_left, bottom_right, (0, 255, 0), 2)
                detected_image = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
                
                # Calculate time
                rectangle_time = (frame_count / frame_rate) * 1000
                logging.info(f"Mobile rectangle detected at {rectangle_time} ms")
                break

        cap.release()

        if rectangle_time is None:
            logging.info("No rectangle detected in mobile video.")
            return None, None

        return rectangle_time, detected_image
    except Exception as e:
        logging.error(f"Failed to find gradient rectangle in mobile video: {str(e)}")
        raise

def extract_beep_audio(audio_path, beep_time, output_path, duration=500):
    """Extract a snippet of the audio around the beep time."""
    try:
        audio = AudioSegment.from_wav(audio_path)
        start_time = beep_time - duration / 2
        end_time = beep_time + duration / 2
        snippet = audio[int(start_time):int(end_time)]
        snippet.export(output_path, format="wav")
        logging.info(f"Beep audio snippet saved at {output_path}")
    except Exception as e:
        logging.error(f"Failed to extract beep audio: {str(e)}")
        raise

bot.run(TOKEN)
