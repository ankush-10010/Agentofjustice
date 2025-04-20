import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Optional: change voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Choose female/male based on index

def speak(text, juror_name="Juror"):
    print(f"{juror_name} says: {text}")
    engine.say(text)
    engine.runAndWait()

speak("Hello, I am your virtual juror. How can I assist you today?")
speak("ioned therein on a turnkey basis. On 25.9.2008, the appellant companypany, which is Signature Not Verified involved in civil electrical works in India, was awarded the said Digitally signed by NIDHI AHUJA Date 2019.03.11 173359 IST Reason tender after having been found to be the best suited for the task. On 16.1.2009, a formal companytract was entered into between the appellant and respondent No. 2. It may be mentioned that the numberice inviting tender formed part and parcel of the formal agreement. Contained in the numberice inviting tender is a detailed arbitration clause. In this matter, we are companycerned with clause 25 viii  which is set out as follows- viii. It shall be an essential term of this companytract that in order to avoid frivolous claims the party invoking arbitration shall specify the dispute based on facts and calculations stating the amount claimed")