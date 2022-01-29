paths = dict(
	dataset 			= 'data/dataset/',
	groundtruth			= 'data/groundtruth/',
	datasignature 	= '../data/datasignature/',
	dataset_upscaled 	= dict(
		edsr	= '../data/dataset_upscaled/edsr/',
		espcn	= '../data/dataset_upscaled/espcn/',
		lapsrn	= '../data/dataset_upscaled/lapsrn/',
		test	= '../data/dataset_upscaled/test/',
	),
	pseudo_labels		= '../data/pseudo_labels/',
	weights 	= dict(
		oxford	= 'data/weights/oxford_segmentation_20_epochs.h5',
		edsr	= 'data/weights/EDSR_x4.pb',
		espcn	= 'data/weights/ESPCN_x4.pb',
		lapsrn	= 'data/weights/LapSRN_x4.pb',
	),
	imagenet_labels = '../data/imagenet_class_index.json'
)

labels = [
	'cat',
	'lynx',
	'wolf',
	'coyote',
	'cheetah',
	'jaguar',
	'chimpanzee',
	'orangutan',
	'hamster',
	'guinea_pig'
]