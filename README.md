# Easy-Wav2Lip: Enhanced Wav2Lip for Video Lip-Syncing

## Contents
1. [Introduction](#introduction)
2. [Google Colab Version (Free In-Browser Cloud Computing)](#google-colab-version)
3. [Local Installation](#local-installation)
4. [Usage](#usage)
5. [Support](#support)
6. [Best Practices](#best-practices)
7. [Advanced Options](#advanced-options)
8. [Batch Processing](#batch-processing)
9. [Credits](#credits)

---

## Introduction
**Easy-Wav2Lip** is a tool designed to simplify and enhance lip-syncing in videos using the Wav2Lip technology. Its key benefits include:

- **Ease of Use:**  
  - Simple setup both locally and via Google Colab.
  - Only 2 cells to execute in Google Colab.
  - Windows users need only a single installation file.
  
- **Speed:**  
  - For a 9-second 720p 60fps test clip on Colab with a T4 GPU:
  
    | Original Wav2Lip | Easy-Wav2Lip |
    | ---------------- | ------------ |
    | 6m 53s           | 56s          |
  
  - Reprocessing the same video takes just 25 seconds by reusing tracking data.

- **Quality:**  
  - Fixes visual issues with lip movements.
  - Offers three quality options:
    - **Fast:** Basic Wav2Lip.
    - **Improved:** Wav2Lip with a feathered mask around the mouth to retain the original resolution of the rest of the face.
    - **Enhanced:** Wav2Lip with a mask plus face upscaling using GFPGAN.

[![Comparison GIF](https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/wav2lipcomparison.gif)](https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/wav2lipcomparison.gif)

---

## Google Colab Version
For the easiest and most compatible experience, we recommend using the Google Colab version:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/overklassniy/Easy-Wav2Lip/blob/v9.0/Easy_Wav2Lip_v9.0.ipynb)

---

## Local Installation

### Requirements
- **Nvidia:** A graphics card supporting CUDA 12.6  
or  
- **MacOS:** A device supporting MPS via Apple Silicon or an AMD GPU

### Automatic Installation for Windows (64-bit and x86)
1. [Download the Easy-Wav2Lip.cmd file](https://github.com/overklassniy/Easy-Wav2Lip/blob/Installers/Easy-Wav2Lip.cmd).
2. Place it in a folder of your choice (e.g., Documents).
3. Run the file and follow the on-screen instructions. A folder named **Easy-Wav2Lip** will be created in the selected directory.
4. Use this file whenever you want to run Easy-Wav2Lip.

### Manual Installation

#### Required Components
- Python 3.13.2 (other versions may not work)
- [Git](https://git-scm.com/)
- For Windows and Linux: CUDA (typically, having the latest Nvidia drivers is sufficient; tested with CUDA 12.6)

#### Windows Manual Installation
1. Open Command Prompt (CMD) and navigate to the folder where you want to install the tool (e.g., `cd Documents`).
2. Execute the following commands (this will create two folders: `Easy-Wav2Lip` and `Easy-Wav2Lip-venv` for an isolated Python environment):

    ```bash
    py -3.13 -m venv EW2L_venv
    EW2L_venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install requests
    python -c "import requests; r = requests.get('https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip', stream=True); open('ffmpeg.zip', 'wb').write(r.content)"
    powershell -Command "Expand-Archive -Path .\\ffmpeg.zip -DestinationPath .\\"
    xcopy /e /i /y "ffmpeg-master-latest-win64-gpl\bin\*" .\EW2L_venv\Scripts
    del ffmpeg.zip
    rmdir /s /q ffmpeg-master-latest-win64-gpl
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    python -m pip install "https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/dlib-19.24.6-cp313-cp313-win_amd64.whl"
    python -m pip install "https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/basicsr-1.4.2-py3-none-any.whl"
    python -m pip install "https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/gfpgan-1.3.8-py3-none-any.whl"
    git clone https://github.com/overklassniy/Easy-Wav2Lip.git
    cd Easy-Wav2Lip
    python -m pip install -r requirements.txt
    python install.py
    ```

3. To run Easy-Wav2Lip, close and reopen CMD, then navigate back to the installation folder:
    ```bash
    Easy-Wav2Lip-venv\Scripts\activate
    cd Easy-Wav2Lip
    python GUI.py
    ```

For further instructions, refer to the [Usage](#usage) section.

---

## Usage
After installation, a file named `config.ini` will open automatically. In this file, you need to:
- Specify the paths to your video and audio files.
- Adjust the settings according to your needs.

**Pro Tips:**
- **Windows:** Hold the Shift key, right-click on the desired file, and select "Copy as path" to easily obtain the file path.
- **MacOS:** Right-click the file while holding the Option (Alt) key and select “Copy [filename] as Pathname”.

Once you save and close `config.ini`, the Wav2Lip process will begin, and the output file will be saved in the same directory as your video file. For additional details, see the [Best Practices](#best-practices) and [Advanced Options](#advanced-options) sections.

---

## Support
If you encounter any issues:
- Check the [Issues](https://github.com/overklassniy/Easy-Wav2Lip/issues) section on GitHub.
- When opening a new issue, please include:
  - The Easy-Wav2Lip version (Google Colab or local installation).
  - Information about the files used.
  - For local installations: whether you used EasyWav2Lip.cmd or manual installation, your operating system, GPU model, GPU driver version, Python version, and details about the files used.

For general discussions about lip-syncing and related topics, join our [Discord server](https://discord.gg/FNZR9ETwKY).

---

## Best Practices

### Video Files
- The video must display a face in every frame; otherwise, Wav2Lip will not work.
- Crop or mask out faces that should not be lip-synced.
- Use h264-encoded .mp4 files (this is the output format).
- For initial tests, use small, short clips (e.g., less than 720p, under 30 seconds, 30fps).

### Audio Files
- Save audio in .wav format with the same duration as the video.
- Note: Approximately 80ms may be trimmed from the processed audio/video, so consider having a slight buffer.
- You can embed the audio into the video and leave `vocal_path` blank, though this might add a few extra seconds to processing.
- Formats like .mp3 are also supported.

---

## Advanced Options

### Quality Options
Choose from one of the three processing modes:
- **Fast:** Basic Wav2Lip.
- **Improved:** Wav2Lip with an additional mask around the mouth to preserve the original resolution of the rest of the face.
- **Enhanced:** Wav2Lip with a mask and subsequent face upscaling using GFPGAN.

### Lip-Sync Mode Comparison:
| Option           | Pros                                                    | Cons                                             |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------ |
| **Wav2Lip**      | Precise lip-syncing, keeps mouth closed when silent     | Rarely, may result in missing teeth              |
| **Wav2Lip_GAN**  | More visually appealing, retains original facial expressions | Less effective at masking lip movements during silence |

It is recommended to start with **Wav2Lip** and switch to **Wav2Lip_GAN** if you notice artifacts.

### Nosmooth
- **Enabled:** Each frame is processed independently—ideal for fast movements but may cause a “shaky” effect.
- **Disabled:** Face positions are smoothed over 5 frames—better for slow movements but may result in offset issues during rapid movements.

### Padding
Padding settings allow you to adjust the crop around the face:
| Direction | Example Value | Effect                              |
| --------- | ------------- | ----------------------------------- |
| **Up (U)**    | -5            | Removes 5 pixels from the top       |
| **Down (D)**  | 10            | Adds 10 pixels to the bottom        |
| **Left (L)**  | 0             | No change                           |
| **Right (R)** | 15            | Adds 15 pixels to the right         |

### Mask
Configure the mask to control how the processed face blends with the original:
- **size:** Enlarges the mask area.
- **feathering:** Adjusts the blend between the mask center and its edges.
- **mouth_tracking:** Updates the mask position to follow the mouth on each frame (adds processing time).
- **debug_mask:** Enables a debug mode where the background turns grayscale and the mask is highlighted in color for easy visualization.

---

## Batch Processing
Automate the processing of multiple video and/or audio files:
- Name files with a numerical suffix (e.g., `Video1.mp4`, `Video2.mp4`).
- The tool will process numbered files sequentially starting from the selected file.
- If numbered video files are paired with a single audio file, each video will use the same audio. The reverse is also true.
  
**Additional Parameters:**
- **output_suffix:** Adds a suffix to output files to prevent overwriting the originals.
- **include_settings_in_suffix:** Appends the settings used (e.g., Quality, Resolution, nosmooth status, padding) to the output filename.
- **preview_input:** Displays the input video/audio before processing to confirm correct file selection.
- **preview_settings:** Renders a single frame at full size to allow you to adjust settings without processing the entire video.

---

## Credits
- [Original Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- The significant speed boost and improved base quality come from [cog-Wav2Lip](https://github.com/devxpy/cog-Wav2Lip)
- Face upscaling using [GFPGAN](https://github.com/TencentARC/GFPGAN) (based on [wav2lip-hq-updated-ESRGAN](https://github.com/GucciFlipFlops1917/wav2lip-hq-updated-ESRGAN))
- Special thanks to AI assistance (Bing Chat/Copilot) for support during development.
- Shoutout to [JustinJohn](https://github.com/justinjohn0306) for the inspiration from the [Wav2Lip_simplified](https://colab.research.google.com/github/justinjohn0306/Wav2Lip/blob/master/Wav2Lip_simplified_v5.ipynb) Colab notebooks.
