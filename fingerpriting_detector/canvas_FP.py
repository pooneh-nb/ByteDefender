from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from difflib import SequenceMatcher


signature_path = str(Path.home().joinpath(Path.home().cwd(), 'fingerpriting_detector/signitures/canvas_signitures.txt'))
canvas_signatures = utilities.read_file_newline_stripped(signature_path)

def is_tracking(api_list: list):

    for api_args in api_list:
        api = api_args[0]
        args = api_args[1]
        
        if api == "CanvasRenderingContext2D.fillText" or api == "OffscreenCanvasRenderingContext2D.fillText":
            value_fp = ""
            for item in args:
                if item["type"] == "STRING":
                    value_fp = item["value"]
                if value_fp and any(SequenceMatcher(None, value_fp, signature).ratio() > 0.6 for signature in canvas_signatures):
                        return True




