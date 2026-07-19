# config.py

VIDEO_PATH = "/home/user/dev/videos/selected/2026-04-22-lorton-d4-rgb-20m-tracking-2hz-hires-rgb4.mp4"

RGB_WIDTH = 3840
RGB_HEIGHT = 2160
RGB_FRAMERATE = "30/1"
RGB_BITRATE = 10000000

RGB_LOWRES_WIDTH = 1920
RGB_LOWRES_HEIGHT = 1080
RGB_LOWRES_BITRATE = 1000000

THERMAL_WIDTH = 640
THERMAL_HEIGHT = 512
THERMAL_BITRATE = 1000000

THERMAL_LOWRES_WIDTH = THERMAL_WIDTH
THERMAL_LOWRES_HEIGHT = THERMAL_HEIGHT
THERMAL_LOWRES_BITRATE = 1000000

RGB = "rgb"
RGB_LOWRES = "rgbl"
THERMAL = "thermal"
THERMAL_LOWRES = "thermall"


def SOCKET(tag):
    return f"/tmp/{tag}_nv.sock"


SOCKETS = {
    RGB: SOCKET(RGB),
    RGB_LOWRES: SOCKET(RGB_LOWRES),
    THERMAL: SOCKET(THERMAL),
    THERMAL_LOWRES: SOCKET(THERMAL_LOWRES),
}

# File test mode: no camera/v4l2 producers.
PRODUCERS = {}


def FILE_FACTORY(width, height, bitrate):
    return f"""
        (
        filesrc location={VIDEO_PATH} !
        qtdemux !
        queue !
        decodebin !
        videoconvert !
        videoscale !
        videorate !
        video/x-raw,width={width},height={height},framerate={RGB_FRAMERATE} !
        x264enc tune=zerolatency speed-preset=veryfast key-int-max=30 bframes=0 bitrate={bitrate // 1000} !
        h264parse config-interval=1 !
        rtph264pay name=pay0 pt=96 config-interval=1
        )
        """


FACTORIES = {
    RGB: FILE_FACTORY(RGB_WIDTH, RGB_HEIGHT, RGB_BITRATE),
    RGB_LOWRES: FILE_FACTORY(RGB_LOWRES_WIDTH, RGB_LOWRES_HEIGHT, RGB_LOWRES_BITRATE),
}
