import numpy as np
import cv2
import moviepy.editor as mpy


main_img_dir = 'CarTurn/'
main_seg_dir = 'CarTurnAnnot/'


last_img_no = 79

# names of the images in string format
image_names = [('0000'+str(n)+'.jpg')[-9:] for n in range(0,last_img_no+1)]


# Add the processed images to the list
images_list = []


for i in range(len(image_names)):

	# Image is a numpy array with shape (800,1920,3)
	image = cv2.imread(main_img_dir+image_names[i])
	height, width, band = image.shape

	# Seg is the segmentation map of the image with shape(800,1920)
	seg = cv2.imread(main_seg_dir +image_names[i].split('.')[0] + '.png', cv2.IMREAD_GRAYSCALE)

	# we know that seg has maps for objects with ID 38. Thus, I changed it into a binary map for 38.
	# seg = (seg == 38)

	image_of_the_back = image * (seg != 38).reshape(height,width,1)
	image_of_the_obj = image * (seg == 38).reshape(height,width,1)

	## decrease the values of red and green channels by 75%.
	image_of_the_obj[:,:,1:3] //= 4

	output_image = image_of_the_back + image_of_the_obj

	# You can use imwrite function to check the results.
	# cv2.imwrite('output_images/' + image_names[i].split('.')[0] + '_Part_1.png', output_image)

	# append the corrected RGB-channeled images to the list
	images_list.append(output_image[:,:,::-1])


clip = mpy.ImageSequenceClip(images_list, fps=25)
clip.write_videofile('part1_video.mp4', codec='mpeg4')