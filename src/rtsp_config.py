from pathlib import Path

VIDEO_PATH = Path(
    "/home/user/dev/videos/selected/2026-04-22-lorton-d4-rgb-20m-tracking-2hz-hires-rgb4.mp4"
)

RGB = "rgb"
RGB_LOWRES = "rgbl"

RGB_WIDTH = 3840
RGB_HEIGHT = 2160
RGB_FRAMERATE = "30/1"
RGB_BITRATE = 10000000

RGB_LOWRES_WIDTH = 1920
RGB_LOWRES_HEIGHT = 1080
RGB_LOWRES_BITRATE = 2000000

PRODUCERS = {}

SOCKETS = {}


def file_source_pipeline(width: int, height: int, bitrate: int) -> str:
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
        x264enc
            tune=zerolatency
            speed-preset=veryfast
            key-int-max=30
            bframes=0
            bitrate={bitrate // 1000} !
        h264parse config-interval=1 !
        rtph264pay name=pay0 pt=96 config-interval=1
        )
        """


FACTORIES = {
    RGB: file_source_pipeline(RGB_WIDTH, RGB_HEIGHT, RGB_BITRATE),
    RGB_LOWRES: file_source_pipeline(RGB_LOWRES_WIDTH, RGB_LOWRES_HEIGHT, RGB_LOWRES_BITRATE),
}
