@echo off
chcp 1251 > nul
REM =============================================================================
REM Easy-Wav2Lip Launcher
REM =============================================================================

REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Set working directory to the script's directory
cd /d "%~dp0"

REM Define log file location
set "LOGFILE=%~dp0installation.log"

REM Log a message with timestamp
call :log "Starting Easy-Wav2Lip launcher..."

REM Display welcome message
echo Welcome to Easy-Wav2Lip!
call :log "Displayed welcome message."

REM =============================================================================
REM Check for NVIDIA GPU and CUDA-capable drivers
REM =============================================================================
:check_GPU
call :log "Checking for NVIDIA GPU..."
wmic path win32_VideoController get name | findstr /C:"NVIDIA" >nul
if errorlevel 1 (
    echo NVIDIA GPU has not been detected - This requires an Nvidia GPU with CUDA support!
    echo If you do have one installed, try updating your drivers.
    call :log "NVIDIA GPU not detected. Exiting."
    pause
    goto :eof
)
call :log "NVIDIA GPU detected."

REM Check if the driver version supports the required CUDA
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv') do set "driver_version=%%i"
set "major_version=%driver_version:~0,3%"
set "major_version=%major_version:.=%"
if not %major_version% gtr 528 (
    echo Your NVIDIA drivers are outdated - please update them before continuing!
    call :log "Driver version %driver_version% is too low. Exiting."
    pause
    goto :eof
)
call :log "NVIDIA driver version (%driver_version%) is sufficient."

REM =============================================================================
REM Check for Easy-Wav2Lip installation folder
REM =============================================================================
if not exist "Easy-Wav2Lip" (
    set "firstinstall=True"
    echo Creating folders "Easy-Wav2Lip" and "EW2L_venv" for initial installation.
    echo Ensure you have around 4GB of free space and be aware the download may take time.
    call :log "Initial installation: folders not found. Prompting user for installation."
    pause
    mkdir "Easy-Wav2Lip"
    mkdir "Easy-Wav2Lip\firstinstall"
    echo Proceeding with installation...
    call :log "Created required directories for installation."
) else (
    set "firstinstall=False"
    call :log "Easy-Wav2Lip directory exists. Skipping first installation steps."
)

REM =============================================================================
REM Check for Python installation (requires Python 3.13)
REM =============================================================================
:check_python
call :log "Checking for Python 3.13..."
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 is not installed. Downloading Python 3.13.2...
    call :log "Python 3.13 not found. Initiating Python installation."
    call :install_python
) else (
    call :log "Python 3.13 detected."
)

REM =============================================================================
REM Activate or Create Virtual Environment
REM =============================================================================
:activate_venv
REM Check if virtual environment is already active
IF NOT defined VIRTUAL_ENV (
    IF EXIST "EW2L_venv\Scripts\activate.bat" (
        call "EW2L_venv\Scripts\activate.bat"
        call :log "Activated existing virtual environment."
    ) ELSE (
        echo Creating virtual environment...
        call :log "Virtual environment not found. Creating a new one."
        py -3.13 -m venv EW2L_venv
        call "EW2L_venv\Scripts\activate.bat"
        call :log "New virtual environment created and activated."
    )
)

REM Ensure virtual environment is activated and update PATH
IF defined VIRTUAL_ENV (
    set "PATH=%~dp0EW2L_venv\Scripts;%PATH%"
    echo Virtual environment running from %VIRTUAL_ENV%
    call :log "Virtual environment activated: %VIRTUAL_ENV%"
) ELSE (
    echo Error: Virtual environment failed to activate.
    call :log "ERROR: Virtual environment activation failed. Exiting."
    pause
    goto :eof
)

REM =============================================================================
REM Update pip and install required Python packages
REM =============================================================================
call :log "Upgrading pip and installing 'requests' module..."
python -m pip install --upgrade pip >nul 2>&1
python -m pip install requests >nul 2>&1

REM =============================================================================
REM Check for Git installation
REM =============================================================================
:check_git
call :log "Checking for Git installation..."
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed. Downloading Git 2.44.0...
    call :log "Git not found. Initiating Git installation."
    call :install_git
) else (
    call :log "Git is installed."
)

REM =============================================================================
REM Check for ffmpeg, ffplay, and ffprobe
REM =============================================================================
:check_ffmpeg
call :log "Checking for ffmpeg..."
where /q ffmpeg
if %errorlevel% neq 0 (
    echo ffmpeg is not detected. Downloading ffmpeg...
    call :log "ffmpeg not found. Initiating ffmpeg installation."
    call :install_ffmpeg
) else (
    call :log "ffmpeg detected."
)

:check_ffplay
call :log "Checking for ffplay..."
where /q ffplay
if %errorlevel% neq 0 (
    echo ffplay is not detected. Downloading ffplay...
    call :log "ffplay not found. Initiating ffmpeg installation (includes ffplay)."
    call :install_ffmpeg
) else (
    call :log "ffplay detected."
)

:check_ffprobe
call :log "Checking for ffprobe..."
where /q ffprobe
if %errorlevel% neq 0 (
    echo ffprobe is not detected. Downloading ffprobe...
    call :log "ffprobe not found. Initiating ffmpeg installation (includes ffprobe)."
    call :install_ffmpeg
) else (
    call :log "ffprobe detected."
)

REM =============================================================================
REM Retrieve Latest Version Information
REM =============================================================================
:check_version
call :log "Fetching the latest version of Easy-Wav2Lip from GitHub..."
REM Retrieve the default branch name using Python and set as latest_version
for /f %%I in ('python -c "import requests; r = requests.get(\"https://api.github.com/repos/overklassniy/Easy-Wav2Lip\", stream=True); print(r.json().get(\"default_branch\", \"\"))"') do set latest_version=%%I
REM Optionally, a specific version can be set:
REM set latest_version=v9.0

REM If this is the first install, mark accordingly
if exist "Easy-Wav2Lip\firstinstall" (
    set "firstinstall=True"
)

REM =============================================================================
REM Verify Installation or Update
REM =============================================================================
:check_install
REM Check if the 'installed' file exists
if not exist "Easy-Wav2Lip\installed" (
    echo Easy-Wav2Lip appears not to be installed correctly. Reinstall now?
    echo You may need up to 7GB of free space.
    call :log "Installation check failed. 'installed' file not found."
    pause
    goto install
)

REM Read installed version from file
set /p installed_version=<"Easy-Wav2Lip\installed"
echo Easy-Wav2Lip %installed_version% installed.
call :log "Installed version: %installed_version%."

REM Compare installed version with the latest version
if "%installed_version%"=="%latest_version%" (
    call :log "Installed version is up-to-date. Launching GUI..."
    cd Easy-Wav2Lip
    python GUI.py
    goto finished
) else (
    echo Version %latest_version% is now available. Do you want to download and install it? (y/n)
    call :log "Update available: Installed version %installed_version%, latest version %latest_version%."
    :user_input_loop
    set /p user_input=Enter y or n: 
    if /i "!user_input!"=="y" (
        call :log "User chose to update."
        call :install
    ) else if /i "!user_input!"=="n" (
        call :log "User chose not to update. Launching GUI..."
        cd Easy-Wav2Lip
        python GUI.py
        goto finished
    ) else (
        echo Invalid input. Please enter y or n.
        goto user_input_loop
    )
)

:finished
call :log "Closing Easy-Wav2Lip."
echo Closing Easy-Wav2Lip...
timeout /t 1 >nul
goto :eof

REM =============================================================================
REM Installation Routine
REM =============================================================================
:install
call :log "Starting installation routine..."

REM Backup existing checkpoints if present
if exist "Easy-Wav2Lip\checkpoints" (
    echo Saving existing checkpoint files...
    call :log "Backing up checkpoints..."
    xcopy /e /i /y "Easy-Wav2Lip\checkpoints" "temp\checkpoints"
)

REM Remove existing Easy-Wav2Lip directory
if exist "Easy-Wav2Lip" (
    call :log "Removing old Easy-Wav2Lip directory..."
    rmdir /s /q "Easy-Wav2Lip"
)

REM Set Git executable path
set "GitPath=%ProgramFiles%\Git\bin\git.exe"
where git >nul 2>&1
if %errorlevel% equ 0 (
    set "GitPath=git"
)
call :log "Using Git executable: %GitPath%."

REM Install additional Python dependencies required before cloning
call :log "Installing prerequisite Python packages..."
python -m pip install https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/dlib-19.24.6-cp313-cp313-win_amd64.whl
python -m pip install https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/basicsr-1.4.2-py3-none-any.whl
python -m pip install https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/gfpgan-1.3.8-py3-none-any.whl

REM Clone the repository using Git
call :log "Cloning Easy-Wav2Lip repository, branch %latest_version%..."
"%GitPath%" clone -b %latest_version% https://github.com/overklassniy/Easy-Wav2Lip.git

REM Restore checkpoints if they were backed up
if exist temp\checkpoints (
    call :log "Restoring checkpoint files..."
    xcopy /e /i /y "temp\checkpoints" "Easy-Wav2Lip\checkpoints"
)
rmdir /s /q temp

REM Install requirements and run installation script
cd Easy-Wav2Lip
IF defined VIRTUAL_ENV (
    echo Virtual environment running from %VIRTUAL_ENV%
)
call :log "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt
call :log "Running installation script..."
python install.py
cd ..
goto :check_install

REM =============================================================================
REM Python Installation Routine
REM =============================================================================
:install_python
set "url=https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe"
set "outfile=%~dp0install_python.exe"
call :log "Downloading Python from %url%..."
powershell -Command "Invoke-WebRequest -Uri %url% -OutFile %outfile%"
python --version >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.13.2 to PATH...
    call :log "Installing Python 3.13.2 with PATH modification."
    start /wait install_python.exe /quiet PrependPath=1
) else (
    echo Installing Python 3.13.2 without PATH modification...
    call :log "Installing Python 3.13.2 without PATH modification."
    start /wait install_python.exe /quiet PrependPath=0
)
del install_python.exe
echo Python installation complete.
call :log "Python installation complete."
goto :eof

REM =============================================================================
REM Git Installation Routine
REM =============================================================================
:install_git
set "url=https://github.com/git-for-windows/git/releases/download/v2.49.0.windows.1/Git-2.49.0-64-bit.exe"
set "outfile=%~dp0install_git.exe"
call :log "Downloading Git from %url%..."
python -c "import requests; r = requests.get('%url%', stream=True); open('%outfile%', 'wb').write(r.content)"
echo Git downloaded, installing...
call :log "Installing Git..."
start /wait install_git.exe /SILENT /NORESTART /NOCANCEL /SP- /LOG
del install_git.exe
echo Git installation complete.
call :log "Git installation complete."
goto :eof

REM =============================================================================
REM ffmpeg Installation Routine
REM =============================================================================
:install_ffmpeg
call :log "Installing ffmpeg (and related tools)..."
call :get_python_path
set "url=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "outfile=%~dp0ffmpeg.zip"
call :log "Downloading ffmpeg from %url%..."
python -c "import requests; r = requests.get('%url%', stream=True); open('%outfile%', 'wb').write(r.content)"
echo ffmpeg downloaded - installing...
powershell -Command "Expand-Archive -Path '%outfile%' -DestinationPath '%~dp0'"
del "%outfile%"
xcopy /e /i /y "ffmpeg-master-latest-win64-gpl\bin\*" "%python_path%"
rmdir /s /q ffmpeg-master-latest-win64-gpl
echo ffmpeg installed.
call :log "ffmpeg installation complete."
goto :eof

REM =============================================================================
REM Helper: Get Python Scripts Path from Virtual Environment
REM =============================================================================
:get_python_path
set "python_path=%VIRTUAL_ENV%\Scripts"
goto :eof

REM =============================================================================
REM Helper: Log Function (logs messages with timestamp)
REM =============================================================================
:log
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set TIMESTAMP=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2% %datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%
    echo [%TIMESTAMP%] %~1 >> "%LOGFILE%"
    goto :eof

REM End of script
:end
pause
endlocal
