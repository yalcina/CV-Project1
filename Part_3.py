import numpy as np
import cv2
import moviepy.editor as mpy



def average(images):
	
	total = images[0].copy()
	
	for i in range(1,len(images)):
		total = total + images[i]
	
	return np.around(total / len(images))



# Images concatenated by width to form single long joined image in order to calculate the average histogram
def concatenate_by_width(images):
	return np.concatenate(images, axis=1)




def mask_images(imgs_list, obj_seg_list, back_seg_list):
	
	# first, each list of images/segment images converted to array of images/segment images
	arr_imgs = np.array(imgs_list)
	arr_obj_seg = np.array(obj_seg_list)
	arr_back_seg = np.array(back_seg_list)

	arr_obj_seg = arr_obj_seg.reshape(arr_obj_seg.shape + (1,))
	arr_back_seg = arr_back_seg.reshape(arr_back_seg.shape + (1,))

	arr_obj_seg = np.concatenate((arr_obj_seg,arr_obj_seg,arr_obj_seg), axis=3)
	arr_back_seg = np.concatenate((arr_back_seg,arr_back_seg,arr_back_seg), axis=3)

	# Object and Background masked according to the segmentation maps
	# (btw filtered regions marked as -1 in order to be ignored in future operations)
	obj_masked_arr = arr_imgs * arr_obj_seg - (arr_obj_seg == False)
	print('\nObject masked')
	back_masked_arr = arr_imgs * arr_back_seg - (arr_back_seg == False)
	print('Background masked\n')

	return obj_masked_arr, back_masked_arr




# extracting single image's histogram
def extract_histogram(image):
	
	img = image.copy()
	height, width, band = img.shape
	
	# vectorizing each channel so that we can compute histogram
	red_vector = img[:,:,2].reshape(height*width)
	green_vector = img[:,:,1].reshape(height*width)
	blue_vector = img[:,:,0].reshape(height*width)
	
	
	# incomplete histograms(some missing values, zeros not appeared)
	intensities_Red, hist_Red = np.unique(red_vector, return_counts=True)
	intensities_Green, hist_Green = np.unique(green_vector, return_counts=True)
	intensities_Blue, hist_Blue = np.unique(blue_vector, return_counts=True)
	
	intensities_Red = intensities_Red.astype(np.int64)
	intensities_Green = intensities_Green.astype(np.int64)
	intensities_Blue = intensities_Blue.astype(np.int64)
	
	
	if intensities_Red[0] < 0:
		intensities_Red = intensities_Red[1:]
		hist_Red = hist_Red[1:]
	
	if intensities_Green[0] < 0:
		intensities_Green = intensities_Green[1:]
		hist_Green = hist_Green[1:]
	
	if intensities_Blue[0] < 0:
		intensities_Blue = intensities_Blue[1:]
		hist_Blue = hist_Blue[1:]
	
	
	R = np.zeros(shape=(256,), dtype=int)
	G = np.zeros(shape=(256,), dtype=int)
	B = np.zeros(shape=(256,), dtype=int)
	
	
	for i,n in enumerate(intensities_Red):
		R[n] = hist_Red[i]
	for i,n in enumerate(intensities_Green):
		G[n] = hist_Green[i]
	for i,n in enumerate(intensities_Blue):
		B[n] = hist_Blue[i]
	
	# Normalization to [0-1]
	R = R / np.sum(R)
	G = G / np.sum(G)
	B = B / np.sum(B)
	return (R,G,B)




def obtain_CDF(hist):
	CDF = hist.copy()
	
	for i in range(1,len(CDF)):
		CDF[i] += CDF[i-1]
	
	CDF[len(CDF)-1] = 1.0
	return CDF




# using list's index for original CDF's intensity and list's values for target CDF's intensity
def create_LUT_for_single_channel(orig_CDF,target_CDF):
	LUT = []
	j = 0
	
	for i in range(len(orig_CDF)):
		while (target_CDF[j] < orig_CDF[i]) and (j < 255):
			j += 1
		LUT.append(j)
	
	return LUT




def transform_channel(img_channel,LUT):
	"""
	This function takes an image channel (only red, green or blue) in 2D array, \
applies the transformation for each pixel (except for the ones with negative intensities, \
because these pixels are maybe intended to be excluded in transformation operation, for example, \
segmented image transformaiton), and then returns the transformed array.
	"""
	def transform_pixel(n):
		if n < 0:
			return 0
		else:
			return LUT[n]
	
	vec_func = np.vectorize(transform_pixel)
	return vec_func(img_channel)





def transform_the_image(img, LUT_red, LUT_green, LUT_blue):
	""" Transforming only 1 image """
	
	red_channel = img[:,:,2].copy()
	green_channel = img[:,:,1].copy()
	blue_channel = img[:,:,0].copy()
	
	transformed_red = transform_channel(red_channel,LUT_red)
	transformed_green = transform_channel(green_channel,LUT_green)
	transformed_blue = transform_channel(blue_channel,LUT_blue)
	
	# shape(3,2160,3840) -->  shape(2160,3840,3)
	im = np.array([transformed_red, transformed_green, transformed_blue])
	return np.swapaxes(np.swapaxes(im,0,1),1,2)




main_img_dir = 'CarTurn/'
main_seg_dir = 'CarTurnAnnot/'
main_targ_dir = 'targets/'

# but sequence of 5 or so images is recommended for an appropriate running time and memory consumption
last_img_no = 79
image_names = [('0000'+str(n)+'.jpg')[-9:] for n in range(0,last_img_no+1)]


# Choose the target images from the file "targets" in whatever order you want
targ_1 = 'target_1.jpg'
targ_2 = 'target_3.jpg'


target_img_1 = cv2.imread(main_targ_dir + targ_1)
target_img_2 = cv2.imread(main_targ_dir + targ_2)

targ_1_hist_Red, targ_1_hist_Green, targ_1_hist_Blue = extract_histogram(target_img_1)
targ_2_hist_Red, targ_2_hist_Green, targ_2_hist_Blue = extract_histogram(target_img_2)


targ_1_CDF_Red = obtain_CDF(targ_1_hist_Red)
targ_1_CDF_Green = obtain_CDF(targ_1_hist_Green)
targ_1_CDF_Blue = obtain_CDF(targ_1_hist_Blue)

targ_2_CDF_Red = obtain_CDF(targ_2_hist_Red)
targ_2_CDF_Green = obtain_CDF(targ_2_hist_Green)
targ_2_CDF_Blue = obtain_CDF(targ_2_hist_Blue)


# Add the processed images to the list
images_list = []

for i in range(len(image_names)):
	
	image = cv2.imread(main_img_dir+image_names[i])
	# Image is a numpy array with shape (800,1920,3)
	
	image = image.astype(np.int16)
	images_list.append(image)


obj_seg_list = []
background_seg_list = []

for i in range(len(image_names)):
	seg = cv2.imread(main_seg_dir +image_names[i].split('.')[0] + '.png', cv2.IMREAD_GRAYSCALE)
	
	background = (seg == 0)
	obj = (seg == 38)
	
	obj_seg_list.append(obj)
	background_seg_list.append(background)



# All image sequences divided into 3 different segmented image lists(i.e., obj, background)
obj_imgs, back_imgs = mask_images(images_list, obj_seg_list, background_seg_list)

concat_obj = concatenate_by_width(obj_imgs)
concat_back = concatenate_by_width(back_imgs)

# EXTRACTING HISTOGRAMS FROM 3 CONCATENATED IMAGES
hist_obj_red, hist_obj_green, hist_obj_blue = extract_histogram(concat_obj)
print('Histogram for Object created')

hist_back_red, hist_back_green, hist_back_blue = extract_histogram(concat_back)
print('Histogram for Background created\n')

del concat_obj, concat_back



CDF_Obj_Red = obtain_CDF(hist_obj_red)
CDF_Obj_Green = obtain_CDF(hist_obj_green)
CDF_Obj_Blue = obtain_CDF(hist_obj_blue)

CDF_Back_Red = obtain_CDF(hist_back_red)
CDF_Back_Green = obtain_CDF(hist_back_green)
CDF_Back_Blue = obtain_CDF(hist_back_blue)

del hist_obj_red, hist_obj_green, hist_obj_blue, hist_back_red, hist_back_green, hist_back_blue
print('CDFs obtained')


LUT_OBJ_RED = create_LUT_for_single_channel(CDF_Obj_Red, targ_1_CDF_Red)
LUT_OBJ_GREEN = create_LUT_for_single_channel(CDF_Obj_Green, targ_1_CDF_Green)
LUT_OBJ_BLUE = create_LUT_for_single_channel(CDF_Obj_Blue, targ_1_CDF_Blue)

LUT_BACK_RED = create_LUT_for_single_channel(CDF_Back_Red, targ_2_CDF_Red)
LUT_BACK_GREEN = create_LUT_for_single_channel(CDF_Back_Green, targ_2_CDF_Green)
LUT_BACK_BLUE = create_LUT_for_single_channel(CDF_Back_Blue, targ_2_CDF_Blue)

print('Look-up Tables(LUT) created\n')
print(f'Image transformation being applied ({last_img_no+1} images):')

transformed_obj_imgs  = []
transformed_back_imgs = []

for i in range(len(images_list)):
	transf_obj = transform_the_image(obj_imgs[i], LUT_OBJ_RED, LUT_OBJ_GREEN, LUT_OBJ_BLUE)
	transformed_obj_imgs.append(transf_obj)
	
	transf_back = transform_the_image(back_imgs[i], LUT_BACK_RED, LUT_BACK_GREEN, LUT_BACK_BLUE)
	transformed_back_imgs.append(transf_back)
	print(f'{i}. image transformed')



transformed_imgs = [obj + background for obj, background in zip(transformed_obj_imgs, transformed_back_imgs)]

clip = mpy.ImageSequenceClip(transformed_imgs, fps=25)
clip.write_videofile('part3_video.mp4', codec='mpeg4')