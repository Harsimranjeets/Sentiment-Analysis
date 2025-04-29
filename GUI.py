import gradio as gr
import emoji
import string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

cyberbullying_vectorizer = joblib.load("final/vectorizer_cyberbullying.pkl")  
cyberbullying_model = joblib.load("final/cyberbullying_model.pkl")

emotion_vectorizer = joblib.load("E:/Minor Project/final/vectorizer_emotion.pkl")
emotion_model = joblib.load("E:/Minor Project/final/model_emotion.pkl")

sentiment_vectorizer = joblib.load("E:/Minor Project/final/vectorizer_sentiment.pkl")
sentiment_model = joblib.load("E:/Minor Project/final/sentiment_model.pkl")

abbreviations = {
        "2B": "To Be",
        "2M": "Too Much",
        "2N": "To Night",
        "2NITE": "Tonight",
        "2U": "To You",
        "UR": "Your",
        "U" : "You",
        "AFAIK": "As Far As I Know",
        "AFK": "Away From Keyboard",
        "ASAP": "As Soon As Possible",
        "ATK": "At The Keyboard",
        "ATM": "At The Moment",
        "A3": "Anytime, Anywhere, Anyplace",
        "BAK": "Back At Keyboard",
        "BBL": "Be Back Later",
        "BBS": "Be Back Soon",
        "BFN": "Bye For Now",
        "B4N": "Bye For Now",
        "BRB": "Be Right Back",
        "BRT": "Be Right There",
        "BTW": "By The Way",
        "B4": "Before",
        "CU": "See You",
        "CUL8R": "See You Later",
        "CYA": "See You",
        "FAQ": "Frequently Asked Questions",
        "FC": "Fingers Crossed",
        "FWIW": "For What It's Worth",
        "FYI": "For Your Information",
        "GAL": "Get A Life",
        "GG": "Good Game",
        "GN": "Good Night",
        "GMTA": "Great Minds Think Alike",
        "GR8": "Great!",
        "G9": "Genius",
        "IC": "I See",
        "ICQ": "I Seek you (also a chat program)",
        "ILU": "I Love You",
        "IMHO": "In My Honest/Humble Opinion",
        "IMO": "In My Opinion",
        "IOW": "In Other Words",
        "IRL": "In Real Life",
        "KISS": "Keep It Simple, Stupid",
        "LDR": "Long Distance Relationship",
        "LMAO": "Laugh My A.. Off",
        "LOL": "Laughing Out Loud",
        "LTNS": "Long Time No See",
        "L8R": "Later",
        "MTE": "My Thoughts Exactly",
        "M8": "Mate",
        "NRN": "No Reply Necessary",
        "OIC": "Oh I See",
        "PITA": "Pain In The A..",
        "PRT": "Party",
        "PRW": "Parents Are Watching",
        "QPSA?": "Que Pasa?",
        "ROFL": "Rolling On The Floor Laughing",
        "ROFLOL": "Rolling On The Floor Laughing Out Loud",
        "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
        "SK8": "Skate",
        "STATS": "Your sex and age",
        "ASL": "Age, Sex, Location",
        "THX": "Thank You",
        "TTFN": "Ta-Ta For Now!",
        "TTYL": "Talk To You Later",
        "U": "You",
        "U2": "You Too",
        "U4E": "Yours For Ever",
        "WB": "Welcome Back",
        "WTF": "What The F...",
        "WTG": "Way To Go!",
        "WUF": "Where Are You From?",
        "W8": "Wait...",
        "7K": "Sick:-D Laugher",
        "TFW": "That feeling when",
        "MFW": "My face when",
        "MRW": "My reaction when",
        "IFYP": "I feel your pain",
        "TNTL": "Trying not to laugh",
        "JK": "Just kidding",
        "IDC": "I don’t care",
        "ILY": "I love you",
        "IMU": "I miss you",
        "ADIH": "Another day in hell",
        "ZZZ": "Sleeping, bored, tired",
        "WYWH": "Wish you were here",
        "TIME": "Tears in my eyes",
        "BAE": "Before anyone else",
        "FIMH": "Forever in my heart",
        "BSAAW": "Big smile and a wink",
        "BWL": "Bursting with laughter",
        "BFF": "Best friends forever",
        "CSL": "Can’t stop laughing"
    }

# Preprocessing functions
def remove_punc(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def replace_abbreviations(text):
    words = text.split()
    result = []
    for word in words:
        upper_word = word.upper()
        if upper_word in abbreviations:
            result.append(abbreviations[upper_word])
        else:
            result.append(word)
    return " ".join(result)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess_text(user_input):
    text = user_input.lower()
    text = emoji.demojize(text)
    text = remove_punc(text)
    text = replace_abbreviations(text)
    text = remove_stopwords(text)
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

# Prediction function
def predict(user_input):
    processed_text = preprocess_text(user_input)

    # Cyberbullying Prediction
    cyber_input = cyberbullying_vectorizer.transform([processed_text])
    cyber_prediction = cyberbullying_model.predict(cyber_input)[0]

    cyber_result = {
        0: "Cyberbullying ", #(ethnicity/race)
        1: "Cyberbullying ",  #(gender/sexual)
        2: "Not Cyberbullying",
        3: "Cyberbullying "  #(religion)
    }.get(cyber_prediction, "Unknown")

    # Emotion Prediction
    emotion_input = emotion_vectorizer.transform([processed_text])
    emotion_prediction = emotion_model.predict(emotion_input)[0]
    emotion_result = {
        0: "Sadness",
        1: "Joy",
        2: "Loved",
        3: "Anger",
        4: "Fear",
        5: "Surprise"
    }.get(emotion_prediction, "Unknown")

    # Sentiment Prediction
    sentiment_input = sentiment_vectorizer.transform([processed_text])
    sentiment_prediction = sentiment_model.predict(sentiment_input)[0]
    sentiment_result = {
        0: "Negative",
        1: "Positive"
    }.get(sentiment_prediction, "Unknown")

    if cyber_prediction != 2 and sentiment_prediction == 0:
        emotion_result = "Anger"
    if sentiment_prediction == 1:
        emotion_result = "joy"
    if cyber_prediction == 2 and emotion_prediction != 3:
       sentiment_result = "Positive"

    return (f"📢 Cyberbullying Status: {cyber_result}\n\n"
            f"❤️ Emotion Detected: {emotion_result}\n\n"
            f"🔵 Sentiment: {sentiment_result}")


# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a message..."),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Decoding Digital Conversation",
    description="Type a message to detect Cyberbullying, Emotion, and Sentiment."
)

interface.launch(share=True)