import speech_recognition as sr

if __name__ == "__main__":

	r = sr.Recognizer()
	m = sr.Microphone()

	def loop():
	    with m as source:
	        print("say something")
	        audio = r.listen(source)
	        try:
	            print("you said "+r.recognize_google(audio))
	        except sr.UnknownValueError:
	            print("Could not understand")
	        except sr.RequestError as e:
	            print("errpr: {0}".format(e))

	loop()