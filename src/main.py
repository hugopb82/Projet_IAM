import argparse
import os
import keras
import random

from src.pl_sequence import PL_Sequence
from src import config

def main(args):
	# Constants
	EPOCHS = 50
	BATCH_SIZE = 32
	STEPS_PER_EPOCH = 50
	NUM_CLASSES = 10
	IMG_SIZE = (64, 64)
	VALIDATION_SAMPLES = 50
	UPSCALING_METHOD = 'espcn'

	# Build the model
	from src.build_model import build_model
	# model = build_model(IMG_SIZE, NUM_CLASSES)
	model = keras.models.load_model('results/bestgc40.h5')
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
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
	callbacks = [
		keras.callbacks.ModelCheckpoint("results/best.h5", save_best_only=True)
	]
	model.fit(train, epochs=EPOCHS, validation_data=validation, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks)
	model.save('results/final.h5')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--upscale', default='espcn',
    	help='Upscaling method : espcn, edsr, lapsrn')
	parser.add_argument('-e', '--epochs', type=int, default=10,
		help='Number of epochs')
	args = parser.parse_args()
	main(args)