import sounddevice as sd
import numpy as np
import whisper
import sys
from queue import Queue
from threading import Thread
import nltk
import scipy.io.wavfile as wavf
import copy
# SETTINGS
MODEL_TYPE="base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
BLOCKSIZE=24678
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD=700
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=100
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
whisper_model = whisper.load_model(MODEL_TYPE)
recordings = Queue()

def inputstream_generator():
	def callback(indata,frames, time, status):
		recordings.put(indata)
	print("inputstream_generator")
	chunk=1024
	frames = []
	round = 0
	with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback):
		count1 = 0
		while True:
			frames = recordings.get()
			wavf.write(str(count1)+"___.wav", 16000, frames.flatten().astype(np.float32) / 32768.0)
			count1+=1

			yield frames

def inputstream_generator22():
	global global_ndarray
	def callback(indata,frames, time, status):
		recordings.put(indata)
	print("inputstream_generator")
	chunk=1024
	frames = []
	round = 0
	with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback):
		count1 = 0
		count = 0
		wait = 0
		while True:
			frames = recordings.get()
			indata_flattened = abs(frames.flatten())
			print(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size)
			if((np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size > SILENCE_RATIO)):
				frame1 = copy.deepcopy(frames)
				new_detected = True
				if (global_ndarray is not None):
					print("Selection a")
					global_ndarray1 = np.concatenate((global_ndarray, frame1), dtype='int16')
					global_ndarray = copy.deepcopy(global_ndarray1)
				else:
					print("Selection b")
					global_ndarray = frame1

				# concatenate buffers if the end of the current buffer is not silent
				if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/2):
					print("Selection 1")
					continue
				else:
					print("Selection 2")
					local_ndarray = global_ndarray.copy()
					global_ndarray = None
					indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
					count+=1
					result = whisper_model.transcribe(indata_transformed, language=LANGUAGE,fp16 = False)
					print(nltk.sent_tokenize(result["text"]))

		del local_ndarray
		del indata_flattened
		time.sleep(1)

def process_audio_buffer():
	global global_ndarray
	count = 0
	while not messages.empty():
		print("Recognition")
		for frames in inputstream_generator():
			wavf.write(str(count)+"__.wav", 16000, frames.flatten().astype(np.float32) / 32768.0)
			indata_flattened = abs(frames.flatten())
			if (global_ndarray is not None):
				global_ndarray = np.concatenate((global_ndarray, frames), dtype='int16')
			else:
				global_ndarray = frames
			print(np.average((indata_flattened[-100:-1])))

			# concatenate buffers if the end of the current buffer is not silent
			if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/4):

				continue
			else:
				local_ndarray = global_ndarray.copy()
				global_ndarray = None
				indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
				out_f = str(count)+'.wav'
				#wavf.write(out_f, 16000, indata_transformed)
				result = whisper_model.transcribe(indata_transformed, language=LANGUAGE,fp16 = False)
				print(nltk.sent_tokenize(result["text"]))
			count+=1
		del local_ndarray
		del indata_flattened
		time.sleep(1)

messages = Queue()
def recordingTask():
    messages.put(True)

    print("Starting...")
    transcribe = Thread(target=inputstream_generator22)
    transcribe.start()

if __name__ == "__main__":
	recordingTask()
