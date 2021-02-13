import numpy as np
import cv2
import moviepy.editor as mpy



def average(images):
	
	total = images[0].copy()
	
	for i in range(1,len(images)):
		total = total + images[i]
	
	return np.around(total / len(images))



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
			return n
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




def correct_negative_intensities(img):
	"""
	This function corrects/converts the negative-valued pixels to 0-intensity pixels (absolute dark).\
	Negative values/intensities may be required at some points where we want to ignore those pixels \
	during an operation/transformation, especially in segmented transformation. To do so, we can differentiate \
	those pixels from the normal ones
	"""
	img[img < 0] = 0


#---------------------------------------------------------------


main_img_dir = 'CarTurn/'
main_seg_dir = 'CarTurnAnnot/'
main_targ_dir = 'targets/'

last_img_no = 79
all_images = [('0000'+str(n)+'.jpg')[-9:] for n in range(0,last_img_no+1)]

targ = 'target_2.jpg'
target_img = cv2.imread(main_targ_dir + targ)

targ_hist_Red, targ_hist_Green, targ_hist_Blue = extract_histogram(target_img)

targ_CDF_Red = obtain_CDF(targ_hist_Red)
targ_CDF_Green = obtain_CDF(targ_hist_Green)
targ_CDF_Blue = obtain_CDF(targ_hist_Blue)



# Add the processed images to the list
images_list = []

for i in range(len(all_images)):
	
	image = cv2.imread(main_img_dir+all_images[i])
	# Image is a numpy array with shape (800,1920,3)
	
	image = image.astype(np.int64)
	images_list.append(image)



# average image of all image sequences in terms of RGB values
avg_img = average(images_list)

avg_hist_Red, avg_hist_Green, avg_hist_Blue = extract_histogram(avg_img)

avg_CDF_Red = obtain_CDF(avg_hist_Red)
avg_CDF_Green = obtain_CDF(avg_hist_Green)
avg_CDF_Blue = obtain_CDF(avg_hist_Blue)

LUT_RED = create_LUT_for_single_channel(avg_CDF_Red, targ_CDF_Red)
LUT_GREEN = create_LUT_for_single_channel(avg_CDF_Green, targ_CDF_Green)
LUT_BLUE = create_LUT_for_single_channel(avg_CDF_Blue, targ_CDF_Blue)


transformed_imgs = []
print(f'Image transformation being applied ({last_img_no+1} images):')

for i,img in enumerate(images_list):
	print(f'{i}. image processed')
	trns = transform_the_image(img, LUT_RED, LUT_GREEN, LUT_BLUE)
	transformed_imgs.append(trns)


clip = mpy.ImageSequenceClip(transformed_imgs, fps=25)
clip.write_videofile('part2_video.mp4', codec='mpeg4')