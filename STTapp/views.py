from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import backend as K
import numpy as np
import tempfile
import os
from django.conf import settings
from pydub import AudioSegment


# Define custom CTC Loss for english language
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# Load model only once
english_model_path = os.path.join(settings.BASE_DIR, "STTapp", "models", "speech_models", "english_model_savedmodel")
hindi_model_path = os.path.join(settings.BASE_DIR, "STTapp", "models", "speech_models", "hindi_model_savedmodel")
english_model = tf.keras.models.load_model(
    english_model_path,
    custom_objects={"CTCLoss": CTCLoss}
)
hindi_model = tf.keras.models.load_model(
    hindi_model_path,
    custom_objects={"CTCLoss": CTCLoss}
)

# english Character mappings
en_characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
en_char_to_num = keras.layers.StringLookup(vocabulary=en_characters, oov_token="")
en_num_to_char = keras.layers.StringLookup(
    vocabulary=en_char_to_num.get_vocabulary(), oov_token="", invert=True
)

# hindi Character mappings
# Raw Hindi characters (with potential duplicates)
raw_chars = "ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह़ऽािीुूृेैोौंःॅॉ्।!?' "

# Remove duplicates while preserving order
hi_characters = []
[hi_characters.append(c) for c in raw_chars if c not in hi_characters]

# Now define the StringLookup layers
hi_char_to_num = keras.layers.StringLookup(vocabulary=hi_characters, oov_token="")

hi_num_to_char = keras.layers.StringLookup(
    vocabulary=hi_char_to_num.get_vocabulary(), oov_token="", invert=True
)


# Spectrogram params
frame_length = 256
frame_step = 160
fft_length = 384


# Create your views here.
def landingpage(request):
    return render(request, 'STTapp/landingpage.html')

def STTpage(request):
    return render(request, 'STTapp/STTpage.html')


@csrf_exempt
def speech_recognition(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        language = request.POST.get('language', 'en')  # Default to English

        # Save the uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            for chunk in audio_file.chunks():
                temp_input.write(chunk)
            temp_input_path = temp_input.name
            
        # Convert to real WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_output_path = temp_output.name


        try:
            # Convert WebM or fake WAV to proper PCM WAV using pydub
            audio_segment = AudioSegment.from_file(temp_input_path)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Optional: standardize format
            audio_segment.export(temp_output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
            
            # Call the appropriate model based on language
            if language == 'hi':
                transcription = recognize_hindi(temp_output_path)
            else:
                transcription = recognize_english(temp_output_path)

            return JsonResponse({'status': 'success', 'transcription': transcription})

        except tf.errors.InvalidArgumentError as e:
            # Handle TensorFlow-specific error
            return JsonResponse({'status': 'error', 'message': str(e)})
        
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            os.remove(temp_output_path)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def preprocess_wav_file(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    mean = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddev = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - mean) / (stddev + 1e-10)

    return tf.expand_dims(spectrogram, 0)

def en_decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0].numpy()
    output_text = tf.strings.reduce_join(en_num_to_char(results)).numpy().decode("utf-8")
    return output_text

def hi_decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0].numpy()
    output_text = tf.strings.reduce_join(hi_num_to_char(results)).numpy().decode("utf-8")
    return output_text

def recognize_hindi(audio_path):
    try:
        # Preprocess and predict
        processed = preprocess_wav_file(audio_path)
        infer = hindi_model.signatures["serving_default"]
        result = infer(input=processed)
        logits = list(result.values())[0].numpy()
        decoded = hi_decode_prediction(logits)
        return decoded
    except Exception as e:
        return str(e)


def recognize_english(audio_path):
    try:
        # Preprocess and predict
        processed = preprocess_wav_file(audio_path)
        infer = english_model.signatures["serving_default"]
        result = infer(input=processed)
        logits = list(result.values())[0].numpy()
        decoded = en_decode_prediction(logits)
        return decoded
    except Exception as e:
        return str(e)
