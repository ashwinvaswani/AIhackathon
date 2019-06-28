import os

if __name__ == "__main__":

	user_input = input()

	thresh = 30
	images = []
	data = user_input.split(' ')

	current_dir = os.path.dirname(os.path.realpath('__file__'))
	current_dir = '/media/ubnutu/Windows/Users/hp/Documents/GitHub/ASL-dataset/asl_alphabet_train'
	for word in data:
		for alphabet in word:
			#print(alphabet)
			for folder_name in os.listdir(current_dir):

				#print(folder_name.lower() + alphabet) 
				if folder_name.lower() == alphabet:
				 	#print("yes")
				 	
				 	cnt = 0
				 	for image_name in os.listdir(current_dir + '/' + folder_name):
				 		if cnt < thresh:
				 			frame = cv2.imread(current_dir + '/' + folder_name + '/' + image_name)
				 			images.append(image_name)
				 			print(image_name)
				 			cnt += 1

	images = []
	for f in os.listdir(dir_path):
	    if f.endswith(ext):
	        images.append(f)