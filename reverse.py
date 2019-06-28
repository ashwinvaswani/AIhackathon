import os
import cv2
import time 
import shutil

if __name__ == "__main__":

	print("Enter a paragraph:")
	user_input = input()
	thresh = 20
	images = []
	data = user_input.split(' ')


	current_dir2 = os.path.dirname(os.path.realpath('__file__'))
	current_dir = '/media/ubnutu/Windows/Users/hp/Documents/GitHub/AIhackathon/20 pic dataset'


	if os.path.isdir(current_dir2 + '/' + 'sentence'):
		shutil.rmtree(current_dir2 + '/' + 'sentence')
	os.mkdir(current_dir2 + '/' + 'sentence')
	cnt2 =0
	for word in data:
		for alphabet in word:
			#print(alphabet)
			for folder_name in os.listdir(current_dir):

				#print(folder_name.lower() + alphabet) 
				if folder_name.lower() == alphabet:
				 	#print("yes")
				 	
				 	cnt = 0
				 	for image_name in os.listdir(current_dir + '/' + folder_name):

				 		if cnt == 0:
				 			image_0 = image_name
				 		if cnt < thresh:
				 			frame = cv2.imread(current_dir + '/' + folder_name + '/' + image_name)
				 			cv2.imwrite(current_dir2 + '/sentence/' + image_name,frame)
				 			images.append(image_name)
				 			#print(image_name)
				 			cnt += 1
		for image_name in os.listdir(current_dir + '/nothing'):
			frame = cv2.imread(current_dir + '/nothing/' + image_name)
			cv2.imwrite(current_dir2 + '/sentence/' + image_name,frame)
			images.append(image_name)
			cnt2 += 1

	frame = cv2.imread(current_dir2 + '/sentence/' + image_0)
	cv2.imshow('video',frame)
	height, width, channels = frame.shape

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter('out', fourcc, 20.0, (width, height))


	for image in images:
	    image_path = current_dir2 + '/sentence/' + image
	    frame = cv2.imread(image_path)
	    time.sleep(0.045)
	    out.write(frame) # Write out frame to video
	    print(image)

	    cv2.imshow('video',frame)
	    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
	        break

	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()

	print("The output video is {}".format('out'))