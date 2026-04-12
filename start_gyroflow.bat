cd C:\work\gyroflow-master
set PATH=C:\Qt\6.7.3\msvc2019_64\bin;%PATH%
set PATH=C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin;%PATH%
set PATH=C:\Users\sunjunli2\Downloads\OpenCL-SDK-v2025.07.23-Win-x64\bin;%PATH%

target\release\gyroflow.exe input.mp4 stabilized.mp4 --method 1 --smoothness 2.0 --fov 0.95 --lens GoPro_HERO6Black_Wide_NO-EIS_4by3.json --max-frames 400
