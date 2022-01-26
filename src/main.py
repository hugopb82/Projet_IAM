import argparse
import os
import keras
import random

from pl_sequence import PL_Sequence
import config

def main(args):
	# Constants
	EPOCHS = 10
	BATCH_SIZE = 32
	NUM_CLASSES = 10
	IMG_SIZE = (160, 160)
	VALIDATION_SAMPLES = 5000
	UPSCALING_METHOD = 'espcn'

	# Build the model
	from build_model import build_model
	model = build_model(IMG_SIZE, NUM_CLASSES)
	model.summary()

	# Load the data
	filenames 	= os.listdir(config.paths['dataset'])
	random.Random(1337).shuffle(filenames)
	train 		= PL_Sequence(
		BATCH_SIZE, IMG_SIZE, filenames[:-VALIDATION_SAMPLES], UPSCALING_METHOD
	)
	validation 	= PL_Sequence(
		BATCH_SIZE, IMG_SIZE, filenames[-VALIDATION_SAMPLES:], UPSCALING_METHOD
	)

	# Train the model
	model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
	# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
	callbacks = [
		keras.callbacks.ModelCheckpoint("../results/final.h5", save_best_only=True)
	]
	model.fit(train, epochs=EPOCHS, validation_data=validation, callbacks=callbacks)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--upscale', default='espcn',
    	help='Upscaling method : espcn, edsr, lapsrn')
	parser.add_argument('-e', '--epochs', type=int, default=10,
		help='Number of epochs')
	args = parser.parse_args()
	main(args)