import os
import re
import time
import logging
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, List
import gradio as gr
import numpy as np
import riva.client

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

tts_language_code_keys = [
    key for key, value in os.environ.items()
    if re.compile(r'TTS_LANGUAGE_CODE_\w+').match(key)
]
tts_language_code_keys.sort()
tts_language_codes = [os.environ[key] for key in tts_language_code_keys]

tts_sample_rate = int(os.environ["TTS_SAMPLE_RATE"])

# Full selection of supported TTS language codes (as of Riva 2.14.0)
# and their corresponding names
supported_tts_languages = {
    "tts_language_code": "tts_language_name",
    "en-US": "English",
    "es-ES": "Spanish",
    "es-US": "Spanish",
    "it-IT": "Italian",
    "de-DE": "German",
    "zh-CN": "Mandarin",
}

# Generate a configuration object containing the TTS language code and associated voice names.
# As of Riva 2.14.0, there is only a male TTS voice available for German TTS.
# Otherwise, the code below could easily be implemented with a list comprehension.

tts_config = []

for lang_code in tts_language_codes:
    if lang_code == "de-DE":
        tts_config.append(
            {
                "tts_language_code": lang_code,
                "tts_voice": f'{supported_tts_languages[lang_code]}-{lang_code[-2:]}.Male-1'
            },
        )
    else:
        tts_config.append(
            {
                "tts_language_code": lang_code,
                "tts_voice": f'{supported_tts_languages[lang_code]}-{lang_code[-2:]}.Female-1'
            },
        )
        tts_config.append(
            {
                "tts_language_code": lang_code,
                "tts_voice": f'{supported_tts_languages[lang_code]}-{lang_code[-2:]}.Male-1'
            },
        )



def text_to_speech(text, voice, enable_tts):
    if not text or not voice or not enable_tts:
        gr.Info("Provide all inputs or select an example")
        return None, gr.update(interactive=False)

    tts_dict = next((item for item in tts_config if item['tts_voice'] == voice), None)

    first_buffer = True
    start_time = time.time()

    # TODO: Gradio Flagging doesn't work with streaming audio ouptut.
    # See https://github.com/gradio-app/gradio/issues/5806
    # TODO: Audio download does not work with streaming audio output.
    # See https://github.com/gradio-app/gradio/issues/6570

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

    tts_client = riva.client.SpeechSynthesisService(auth)

    response = tts_client.synthesize_online(
        text=text,
        voice_name=voice,
        language_code=tts_dict['tts_language_code'],
        sample_rate_hz=tts_sample_rate
    )
    for result in response:
        if len(result.audio):
            if first_buffer:
                _LOGGER.info(
                    f"TTS request [{result.id.value}] first buffer latency: {time.time() - start_time} sec"
                )
                first_buffer = False
            yield (tts_sample_rate, np.frombuffer(result.audio, dtype=np.int16))

    _LOGGER.info(f"TTS request [{result.id.value}] last buffer latency: {time.time() - start_time} sec")
    
    yield (tts_sample_rate, np.frombuffer(b'', dtype=np.int16))
