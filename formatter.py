def ljspeech(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"

    # Path to the audio files within the 'wav' subdirectory
    audio_directory = "/content/drive/MyDrive/rdr2_google/3_hour_chunks/wavs"

    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            # Use the filename directly from the metadata, no need to modify
            filename = cols[0].strip()  # Clean up any accidental spaces or quotes

            # Construct the full path to the audio file in the 'wav' subdirectory
            wav_file = os.path.join(audio_directory, filename)

            # Get the transcription text
            text = cols[2].strip()  # Strip any surrounding spaces from the transcription

            # Append the sample
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})

    return items
