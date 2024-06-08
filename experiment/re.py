import re

mean_volume_re = re.compile(r'mean_volume: (?P<mean_volume>-?\d+(\.\d+)?) dB')
max_volume_re = re.compile(r'max_volume: (?P<max_volume>-?\d+(\.\d+)?) dB')
histogram_re = re.compile(r'histogram_(?P<db_level>\d+)db: (?P<count>\d+)')

# Sample data
data = """
+ ffmpeg -i ../input-video/mine.mkv -filter_complex [0]volumedetect[s0] ...

+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    + CategoryInfo          : NotSpecified: (ffmpeg version ...mpeg developers:String) [], RemoteException

    + FullyQualifiedErrorId : NativeCommandError



  built with gcc 12.1.0 (Rev2, Built by MSYS2 project)

  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-lzma

--enable-zlib --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-sdl2 --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxvid --enable-libaom --enable-libopenjpeg

--enable-libvpx --enable-libass --enable-libfreetype --enable-libfribidi --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-ffnvcodec --enable-nvdec

--enable-nvenc --enable-d3d11va --enable-dxva2 --enable-libmfx --enable-libgme --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libtheora --enable-libvo-amrwbenc --enable-libgsm

--enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-librubberband

  libavutil      57. 28.100 / 57. 28.100

  libavcodec     59. 37.100 / 59. 37.100

  libavformat    59. 27.100 / 59. 27.100

  libavdevice    59.  7.100 / 59.  7.100

  libavfilter     8. 44.100 /  8. 44.100

  libswscale      6.  7.100 /  6.  7.100

  libswresample   4.  7.100 /  4.  7.100

  libpostproc    56.  6.100 / 56.  6.100

Input #0, matroska,webm, from '../input-video/mine.mkv':

  Metadata:

    ENCODER         : Lavf58.29.100

  Duration: 00:21:54.57, start: 0.000000, bitrate: 37300 kb/s

  Stream #0:0: Video: h264 (High), yuv420p(tv, smpte170m, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 60 fps, 60 tbr, 1k tbn (default)

    Metadata:

      DURATION        : 00:21:54.567000000

  Stream #0:1: Audio: aac (LC), 48000 Hz, stereo, fltp (default)

    Metadata:

      title           : simple_aac_recording

      DURATION        : 00:21:54.496000000

[Parsed_volumedetect_0 @ 000001ecb01f1dc0] n_samples: 0

Stream mapping:

  Stream #0:1 (aac) -> volumedetect:default

  volumedetect:default -> Stream #0:0 (pcm_s16le)

Press [q] to stop, [?] for help

Output #0, null, to 'pipe:':

  Metadata:

    encoder         : Lavf59.27.100

  Stream #0:0: Audio: pcm_s16le, 48000 Hz, stereo, s16, 1536 kb/s

    Metadata:

      encoder         : Lavc59.37.100 pcm_s16le

size=N/A time=00:00:00.02 bitrate=N/A speed= 167x

size=N/A time=00:05:35.18 bitrate=N/A speed= 670x

size=N/A time=00:11:09.09 bitrate=N/A speed= 669x

size=N/A time=00:16:47.06 bitrate=N/A speed= 671x

size=N/A time=00:21:54.51 bitrate=N/A speed= 660x

video:0kB audio:246472kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown

[Parsed_volumedetect_0 @ 000001ecb031b340] n_samples: 126193664

[Parsed_volumedetect_0 @ 000001ecb031b340] mean_volume: -39.3 dB

[Parsed_volumedetect_0 @ 000001ecb031b340] max_volume: -9.5 dB

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_9db: 12

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_10db: 27

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_11db: 37

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_12db: 99

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_13db: 236

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_14db: 498

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_15db: 1270

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_16db: 5328

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_17db: 21073

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_18db: 33330

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_19db: 46918

[Parsed_volumedetect_0 @ 000001ecb031b340] histogram_20db: 65733

"""

data = "[Parsed_volumedetect_0 @ 000001ecb031b340] mean_volume: -39.3 dB"

# Find matches
mean_volume_match = mean_volume_re.search(data)
max_volume_match = max_volume_re.search(data)
histogram_matches = histogram_re.findall(data)


if mean_volume_match:
    print("mean_volume:", mean_volume_match.group('mean_volume'))

if max_volume_match:
    print("max_volume:", max_volume_match.group('max_volume'))

for match in histogram_matches:
    db_level, count = match
    print(f"histogram_{db_level}db: {count}")


