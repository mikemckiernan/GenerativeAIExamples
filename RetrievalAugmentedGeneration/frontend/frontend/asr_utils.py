import queue
from threading import Thread

import os
import re
import logging
import grpc
import numpy as np
import riva.client
import riva.client.proto.riva_asr_pb2 as riva_asr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from google.protobuf import text_format

class ASRSession:
    def __init__(self):
        self.is_first_buffer = True
        self.request_queue = None
        self.response_stream = None
        self.response_thread = None
        self.transcript = ""

_LOGGER = logging.getLogger(__name__)

# Extract environmental variables
RIVA_SPEECH_API_URI = os.getenv("RIVA_SPEECH_API_URI", None)
NVCF_RIVA_SPEECH_API_URI = os.getenv("NVCF_RIVA_SPEECH_API_URI", None)
NVCF_RUN_KEY = os.getenv("NVCF_RUN_KEY", None)
NVCF_RIVA_FUNCTION_ID = os.getenv("NVCF_RIVA_FUNCTION_ID", None)
if ((RIVA_SPEECH_API_URI is None or RIVA_SPEECH_API_URI == "") and 
    (NVCF_RIVA_SPEECH_API_URI is None or NVCF_RIVA_SPEECH_API_URI == "")):
    _LOGGER.info('At least one of RIVA_SPEECH_API_URI and NVCF_RIVA_SPEECH_API_URI must be set')
if ((NVCF_RIVA_SPEECH_API_URI is not None and NVCF_RIVA_SPEECH_API_URI != "") and
    ((NVCF_RUN_KEY is None or NVCF_RUN_KEY == "") or 
        (NVCF_RIVA_FUNCTION_ID is None or NVCF_RIVA_FUNCTION_ID == ""))):
    _LOGGER.info('If NVCF_RIVA_SPEECH_API_URI is set, NVCF_RUN_KEY and NVCF_RIVA_FUNCTION_ID must also be set')

asr_language_code_keys = [
    key for key, value in os.environ.items() 
    if re.compile(r'ASR_LANGUAGE_CODE_\w+').match(key)
]
asr_language_code_keys.sort()
asr_language_codes = [os.environ[key] for key in asr_language_code_keys]

asr_acoustic_model = os.environ["ASR_ACOUSTIC_MODEL"]

# Full selection of supported ASR language codes (as of Riva 2.14.0)  
# and their corresponding names
supported_asr_languages = {
    "asr_language_code": "asr_language_name",
    "ar-AR": "Armenian", 
    "en-US": "English (US)", 
    "en-GB": "English (UK)", 
    "de-DE": "German", 
    "es-ES": "Spanish (Spain)", 
    "es-US": "Spanish (LatAm)", 
    "fr-FR": "French", 
    "hi-IN": "Hindi",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "ru-RU": "Russian",
    "ko-KR": "Korean",
    "pt-BR": "Portuguese (Brazil)",
    "zh-CN": "Mandarin (China)",
    "es-en-US": "Bilingual Spanish-English (US)",
    "ja-en-JP": "Bilingual Japanese-English (Japan)",
}

# Generate a configuration object containing the human-readable language name and 
# streaming ASR model name associated with each user-specified ASR language code
asr_config = [
    {
        "asr_language_code": lang_code, 
        "asr_language_name": supported_asr_languages[lang_code], 
        "asr_streaming_model_name": f"{asr_acoustic_model}-{lang_code}-asr-streaming"
    }
    for lang_code in asr_language_codes
]

# Establish a connection to the Riva server
try:
    if NVCF_RIVA_SPEECH_API_URI is not None and NVCF_RIVA_SPEECH_API_URI != "":
        metadata = [
            ("authorization", "Bearer " + NVCF_RUN_KEY), 
            ("function-id", NVCF_RIVA_FUNCTION_ID)
        ]
        auth = riva.client.Auth(
            None, use_ssl=True, 
            uri=NVCF_RIVA_SPEECH_API_URI, 
            metadata_args=metadata
        )
    else:
        auth = riva.client.Auth(uri=RIVA_SPEECH_API_URI)
    _LOGGER.info('Created riva.client.Auth success')
except:
    _LOGGER.info('Error creating riva.client.Auth')

def print_streaming_response(asr_session):
    asr_session.transcript = ""
    final_transcript = ""
    try:
        for response in asr_session.response_stream:
            final = ""
            partial = ""
            if not response.results:
                continue
            if len(response.results) > 0 and len(response.results[0].alternatives) > 0:
                for result in response.results:
                    if result.is_final:
                        final += result.alternatives[0].transcript
                    else:
                        partial += result.alternatives[0].transcript

                final_transcript += final
                asr_session.transcript = final_transcript + partial

    except grpc.RpcError as rpc_error:
        _LOGGER.error(rpc_error.code(), rpc_error.details())
        # TODO See if Gradio popup error mechanism can be used.
        # For now whow error via transcript text box.
        asr_session.transcript = rpc_error.details()
        return

# For now, remove second output for 
# protobuf text box
def start_recording(audio, language, asr_session):
    _LOGGER.info('start_recording')
    asr_session.is_first_buffer = True
    asr_session.request_queue = queue.Queue()
    return "", asr_session

def stop_recording(asr_session):
    _LOGGER.info('stop_recording')
    asr_session.request_queue.put(None)
    asr_session.response_thread.join()
    return asr_session

def transcribe_streaming(audio, language, asr_session):
    _LOGGER.info('transcribe_streaming')
    rate, data = audio
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if not len(data):
        return asr_session.transcript, asr_session

    if asr_session.is_first_buffer:
        asr_dict = next((d for d in asr_config if d['asr_language_name'] == language), None)

        streaming_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=asr_dict['asr_language_code'],
                max_alternatives=1,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
                sample_rate_hertz=rate,
                audio_channel_count=1,
                enable_word_time_offsets=True,
                model=asr_dict['asr_streaming_model_name'],
            ),
            interim_results=True,
        )

        _LOGGER.info(f'auth.channel = {auth.channel}')
        rasr_stub = rasr_srv.RivaSpeechRecognitionStub(auth.channel)
        asr_session.response_stream = rasr_stub.StreamingRecognize(iter(asr_session.request_queue.get, None))

        # First buffer should contain only the config
        request = riva_asr.StreamingRecognizeRequest(streaming_config=streaming_config)
        asr_session.request_queue.put(request)

        asr_session.response_thread = Thread(target=print_streaming_response, args=(asr_session,))

        # run the thread
        asr_session.response_thread.start()

        asr_session.is_first_buffer = False

    request = riva_asr.StreamingRecognizeRequest(audio_content=data.astype(np.int16).tobytes())
    asr_session.request_queue.put(request)

    return asr_session.transcript, asr_session

def transcribe_offline(audio, language, diarization):
    _LOGGER.info('transcribe_offline')
    rate, data = audio
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if not len(data):
        _LOGGER.info("Empty audio provided")
        return None, None

    asr_dict = next((d for d in asr_config if d['asr_language_name'] == language), None)

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=rate,
        audio_channel_count=1,
        language_code=asr_dict['asr_language_code'],
        max_alternatives=1,
        profanity_filter=False,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
        enable_word_time_offsets=True,
    )
    riva.client.add_speaker_diarization_to_config(config, diarization)

    asr_client = riva.client.ASRService(auth)
    try:
        response = asr_client.offline_recognize(data.astype(np.int16).tobytes(), config)
        if len(response.results) > 0 and len(response.results[0].alternatives) > 0:
            final_transcript = ""
            for res in response.results:
                final_transcript += res.alternatives[0].transcript
            return final_transcript, text_format.MessageToString(response, as_utf8=True)
    except grpc.RpcError as rpc_error:
        _LOGGER.info(f"{rpc_error.code()}, {rpc_error.details()}")
        # TODO See if Gradio popup error mechanism can be used.
        # For now whow error via transcript text box.
        latest_transcript = rpc_error.details()
        return latest_transcript, None

    return latest_transcript, None