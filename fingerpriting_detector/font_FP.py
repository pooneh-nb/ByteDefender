from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())


def is_tracking(api_list: list):
    unique_fonts = set()
    measure_text_counter = 0

    for api_args in api_list:
        api = api_args[0]
        args = api_args[1]
        
        if api == "CanvasRenderingContext2D.measureText" or api == "OffscreenCanvasRenderingContext2D.measureText":
            measure_text_counter += 1
        if api == "CanvasRenderingContext2D.font.set" or api == "OffscreenCanvasRenderingContext2D.font.set":
            value_fp = ""
            for item in args:
                if item["type"] == "STRING":
                    value_fp = item["value"]
                    unique_fonts.add(value_fp)

        if len(unique_fonts) > 20 and measure_text_counter > 20:
            return True


   

