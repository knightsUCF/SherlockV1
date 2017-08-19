import pyttsx3, time

class Talkamaton():
   
    def __init__(self):
        self.talkamaton = pyttsx3.init()
    
    def Say(self, text):
        self.talkamaton.say(text)
        self.talkamaton.runAndWait()




