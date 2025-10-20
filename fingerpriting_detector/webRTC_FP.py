from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())


RTC_CREATION_APIS = [
    'RTCPeerConnection.createDataChannel',
    'RTCPeerConnection.createOffer']

RTC_EXTRACTION_APIS = [
    'RTCPeerConnection.setLocalDescription',

]

def is_tracking(api_list: list):

    _RTC_CREATION_APIS = False
    _RTC_EXTRACTION_APIS = False
    
    for api_args in api_list:   
        api = api_args[0]
        args = api_args[1]

        if api.startswith('RTCPeerConnection'):
            if any(api == audio_action for audio_action in RTC_CREATION_APIS):
                _RTC_CREATION_APIS = True

            if any(api == audi_extraction for audi_extraction in RTC_EXTRACTION_APIS):
                _RTC_EXTRACTION_APIS = True
            

        if _RTC_CREATION_APIS and _RTC_EXTRACTION_APIS:
            return True


   

