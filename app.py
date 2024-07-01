import streamlit as st
import nltk
import io
import tempfile
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from nltk.tokenize import word_tokenize
import string
import time

nltk.download('punkt')  # to tokenize
nltk.download('stopwords')  # to remove stopwords
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Function to perform sentiment analysis
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative sentiment"
    elif pos > neg:
        return "Positive sentiment"
    else:
        return "Neutral sentiment"

# Load emotions from the file into a dictionary
emotions = {}
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        emotions[word] = emotion.capitalize()  # Capitalize the first letter

# load model
emotion_dict = {0: 'angry', 1:'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

# load json and create model
json_file = open('face_emotion.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into a new model
classifier.load_weights("face_emotion.h5")

# load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


# record audio
def record_audio():
    recognizer = sr.Recognizer()
    text = ""

    with sr.Microphone(device_index=1) as source:
        st.write('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=3)
        st.write('Start Speaking...')

        start_time = time.time()  # Record start time
        recordedAudio = recognizer.listen(source)
        end_time = time.time()  # Record end time

        st.write(f'Done recording! Time taken: {round(end_time - start_time, 2)} seconds')

    # exception handling to throw errors if the program goes wrong
    try:
        st.write('Printing the message...')
        text = recognizer.recognize_google(recordedAudio, language='en-US')
        st.write('Your message: {}'.format(text)) 
    except Exception as ex:
        st.write(ex)

    return text


def recognize_audio(uploaded_audio):
    recognizer = sr.Recognizer()
    text = ""

    # Convert the audio file to a common format (e.g., PCM WAV)
    audio_file = io.BytesIO(uploaded_audio.read())
    audio = AudioSegment.from_file(audio_file, format="wav")

    # Create a temporary file to store the converted audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio.export(temp_audio_file.name, format="wav")

    with sr.AudioFile(temp_audio_file.name) as source:
        st.write('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=3)
        st.write('Analysing uploaded wav file...')
        recordedAudio = recognizer.record(source)

    duration_seconds = len(audio) / 1000  # Duration in seconds
    st.write(f'Done! Duration: {duration_seconds} seconds')

    # exception handling to throw errors if the program goes wrong
    try:
        st.write('Printing the message...')
        text = recognizer.recognize_google(recordedAudio, language='en-US')
        st.write('Your message: {}'.format(text)) 
    except Exception as ex:
        st.write(ex)

    return text


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout.capitalize())  # Capitalize the first letter
            label_position = (x, y)
            # Draw a filled rectangle with a black background
            cv2.rectangle(img, (x, y - 25), (x + w, y), (0, 0, 0), -1)

            # Put white text on the black rectangle
            cv2.putText(img, output, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Sidebar for choosing analysis type
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", ["Text Analysis", "Image Analysis", "Audio Analysis", "Video Analysis"])

    if analysis_type == "Text Analysis":
        st.write("Enter a text and get the predicted Sentiment & Emotions")

        user_input = st.text_area("Text", height=4)

        # Add a "Predict" button
        predict_button = st.button("Predict")

        if predict_button:
            if user_input:
                st.write("Input Text: ", user_input)
                cleaned_text = user_input.lower().translate(str.maketrans('', '', string.punctuation))

                # Tokenize and remove stopwords
                tokenized_words = word_tokenize(cleaned_text, "english")
                final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

                # Get emotions from the user input
                detected_emotions_set = set()  # Use a set to store unique emotions
                for word in final_words:
                    if word in emotions:
                        detected_emotions_set.add(emotions[word])
                
                # Sort the detected emotions
                detected_emotions = sorted(detected_emotions_set)

                # Get sentiment analysis result
                sentiment_result = sentiment_analyse(cleaned_text)

                # Display the results using st.success
                st.success(f"Detected Sentiment: {sentiment_result}")

                if detected_emotions:
                    st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions))
                else:
                    st.info("No emotions detected.")
            else:
                st.warning("Please enter some text.")


    elif analysis_type == "Image Analysis":
        st.write("Upload an Image and get the predicted Emotion")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # Use PIL to open the image
            pil_image = Image.open(uploaded_file)
            original_image = np.array(pil_image)

            # Display the uploaded image without any color changes
            st.image(original_image, use_column_width=True)

            # Add a "Predict" button
            predict_button = st.button("Predict")

            if predict_button:
                # Check if the image is already grayscale
                if original_image.ndim == 2:
                    image = original_image
                else:
                    # Convert the image to grayscale
                    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

                if len(faces) == 0:
                    st.error("No human face detected. Please upload an image containing a human face.")
                # Resize the image to match the input size of the model
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img=original_image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
                        face_image = image[y:y+h, x:x+w]

                        # Resize the image to match the input size of the model
                        face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)

                        # Normalize the image
                        face_image = face_image.astype('float') / 255.0
                        face_image = img_to_array(face_image)
                        face_image = np.expand_dims(face_image, axis=0)

                        # Use the pre-trained model to predict emotions
                        prediction = classifier.predict(face_image)[0]
                        max_index = int(np.argmax(prediction))
                        predicted_emotion = emotion_dict[max_index]

                        # Draw a filled rectangle with a black background
                        cv2.rectangle(original_image, (x, y - 25), (x + w, y), (0, 0, 255), -1)

                        # Put white text on the black rectangle
                        cv2.putText(original_image, predicted_emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display the image with detected faces and predicted emotions
                    st.image(original_image, use_column_width=True)


    elif analysis_type == "Audio Analysis": 
        st.header("Audio Sentiments Analysis using both file upload and Recording")
        # Choose between audio upload or recording
        audio_option = st.selectbox("Choose Audio Option", ["Upload Audio", "Record Audio"])

        if audio_option == "Upload Audio":
            uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

            if uploaded_audio:
                st.audio(uploaded_audio, format="audio/wav")

                # Add a "Predict" button
                predict_button = st.button("Predict")

                if predict_button:
                    audio_text = recognize_audio(uploaded_audio)
                    cleaned_text = audio_text.lower().translate(str.maketrans('', '', string.punctuation))

                    # Tokenize and remove stopwords
                    tokenized_words = word_tokenize(cleaned_text, "english")
                    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

                    # Get emotions from the user input
                    detected_emotions_set = set()  # Use a set to store unique emotions
                    for word in final_words:
                        if word in emotions:
                            detected_emotions_set.add(emotions[word])
                    
                    # Sort the detected emotions
                    detected_emotions = sorted(detected_emotions_set)

                    # Get sentiment analysis result
                    sentiment_result = sentiment_analyse(cleaned_text)

                    # Display the results using st.success
                    st.success(f"Detected Sentiment: {sentiment_result}")

                    if detected_emotions:
                        st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions))
                    else:
                        st.info("No emotions detected.")

        elif audio_option == "Record Audio":
            st.write("Record an Audio and get the predicted Sentiment & Emotions")
            start_recording_button = st.button("Start Recording")

            if start_recording_button:
                audio_text = record_audio()
                cleaned_text = audio_text.lower().translate(str.maketrans('', '', string.punctuation))

                # Tokenize and remove stopwords
                tokenized_words = word_tokenize(cleaned_text, "english")
                final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

                # Get emotions from the user input
                detected_emotions_set = set()  # Use a set to store unique emotions
                for word in final_words:
                    if word in emotions:
                        detected_emotions_set.add(emotions[word])
                
                # Sort the detected emotions
                detected_emotions = sorted(detected_emotions_set)

                # Get sentiment analysis result
                sentiment_result = sentiment_analyse(cleaned_text)

                # Display the results using st.success
                st.success(f"Detected Sentiment: {sentiment_result}")

                if detected_emotions:
                    st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions))
                else:
                    st.info("No emotions detected.")
     
        
    elif analysis_type == "Video Analysis":
        st.header("Webcam Live Feed")
        st.write("Click on start to use the webcam and detect your Face Emotion")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

if __name__ == '__main__':
    main()
