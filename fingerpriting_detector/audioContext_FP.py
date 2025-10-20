from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())


AUDIO_ACTION_APIS = [
    'BaseAudioContext.createOscillator',
    'BaseAudioContext.createDynamicsCompressor',
    'OfflineAudioContext.startRendering',
    'AudioNode.connect',
    'OfflineAudioContext.createOscillator',
    'OfflineAudioContext.createDynamicsCompressor',
    'OfflineAudioContext.destination',
    'OfflineAudioContext.oncomplete']

AUDIO_EXTRACTION_APIS = [
    'AudioBuffer.getChannelData'
]

def is_tracking(api_list: list):
    _AUDIO_ACTION_API = False
    _AUDIO_EXTRACTION_API = False

    for api_args in api_list:
        api = api_args[0]
        args = api_args[1]
        
        if any(api == audio_action for audio_action in AUDIO_ACTION_APIS):
            _AUDIO_ACTION_API = True

        if any(api == audi_extraction for audi_extraction in AUDIO_EXTRACTION_APIS):
            _AUDIO_EXTRACTION_API = True
            
        if _AUDIO_ACTION_API and _AUDIO_EXTRACTION_API:
            return True


   

