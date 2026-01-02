import os
import sys
import uuid
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from nltk.tokenize import wordpunct_tokenize
from nltk import FreqDist
from flask import Flask, render_template, session, request, flash
import speech_recognition as sr
from werkzeug.utils import secure_filename
import tempfile
from library.speech_emotion_recognition import speechEmotionRecognition
from library.text_emotion_recognition import Predict
from library.text_preprocessor import NLTKPreprocessor
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "")
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
name = ""
duration = 0
job_position = ""
text = ""

# Helper functions
def get_personality(text):
    try:
        pred = Predict().run(text, model_name="Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts

def save_text_analysis_data(text, probas, traits):
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = pd.concat([df_text, pd.DataFrame([probas], columns=traits)], ignore_index=True)
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)

    perso = dict(zip(traits, probas))
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index').reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)

    means = {trait: np.mean(df_new[trait]) for trait in traits}
    probas_others = [int(np.mean(df_new[trait]) * 100) for trait in traits]
    df_mean = pd.DataFrame.from_dict(means, orient='index').reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.iloc[df_mean['Value'].idxmax()]['Trait']

    return probas_others, trait_others

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/platform', methods=['POST', 'GET'])
def platform():
    return render_template('platform.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/blog_content')
def blogs_content():
    return render_template('blogs_content.html')

# Interview Scorer
@app.route('/score', methods=['POST', 'GET'])
def score():
    return render_template('scorehome.html')

@app.route('/interview_text', methods=['POST', 'GET'])
def interview_text():
    global duration, name, job_position
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        job_position = request.form['position']
        duration = int(request.form['duration'])
    flash(f"Answer the question in {duration} seconds.")
    return render_template('interview_text.html')

@app.route('/interview', methods=['POST', 'GET'])
def interview():
    global text
    text = request.form.get('answer', '')
    flash(f"After pressing the button, you have {duration} seconds to answer.")
    return render_template('interview.html', name=name, display_button=False, color='#C7EEFF')

@app.route('/audio_recording_interview', methods=['POST', 'GET'])
def audio_recording_interview():
    global duration, text
    SER = speechEmotionRecognition()
    rec_sub_dir = Path('tmp') / 'voice_recording.wav'

    try:
        SER.voice_recording(str(rec_sub_dir), duration=duration)
        flash("Recording over! Evaluate your answer based on emotions expressed.")
        
        r = sr.Recognizer()
        with sr.AudioFile(str(rec_sub_dir)) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            try:
                text += r.recognize_google(audio, key=GOOGLE_KEY)
            except sr.UnknownValueError:
                flash("Could not understand the audio.")
            except sr.RequestError as e:
                flash(f"Speech recognition error: {e}")
    except Exception as e:
        flash(f"Error during recording: {e}")

    return render_template('interview.html', display_button=True, name=name, text=text, color='#00ffad')

@app.route('/interview_analysis', methods=['POST', 'GET'])
def interview_analysis():
    global text, name, job_position
    model_sub_dir = Path('Models') / 'audio.hdf5'
    rec_sub_dir = Path('tmp') / 'voice_recording.wav'
    SER = speechEmotionRecognition(str(model_sub_dir))
    
    step = 1
    sample_rate = 16000
    emotions, timestamp = SER.predict_emotion_from_file(str(rec_sub_dir), chunk_step=step * sample_rate)
    
    SER.prediction_to_csv(emotions, Path("static/js/db") / "audio_emotions.txt", mode='w')
    SER.prediction_to_csv(emotions, Path("static/js/db") / "audio_emotions_other.txt", mode='a')
    
    major_emotion = max(set(emotions), key=emotions.count)
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(Path('static/js/db') / 'audio_emotions_dist.txt', sep=',')
    
    df_other = pd.read_csv(Path("static/js/db") / "audio_emotions_other.txt", sep=",")
    major_emotion_other = df_other.EMOTION.mode()[0]
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION == emotion]) / len(df_other)) for emotion in SER._emotion.values()]
    
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(Path('static/js/db') / 'audio_emotions_dist_other.txt', sep=',')
    
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    probas_others, trait_others = save_text_analysis_data(text, probas, traits)
    
    probas = [int(e * 100) for e in probas]
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['probas'] = probas
    session['text_info'] = {"common_words": [common_words], "num_words": [num_words]}
    
    trait = traits[probas.index(max(probas))]
    
    with open(Path("static/js/db") / "words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for line in counts:
            d.write(f"{line},{counts[line]}\n")
    
    with open(Path("static/js/db") / "words_common.txt", "a") as d:
        for line in counts:
            d.write(f"{line},{counts[line]}\n")
    
    df_words_co = pd.read_csv(Path('static/js/db') / 'words_common.txt', sep=',', on_bad_lines='skip')
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric, errors='coerce')
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv(Path('static/js/db') / 'words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    df_words_perso = pd.read_csv(Path('static/js/db') / 'words_perso.txt', sep=',', on_bad_lines='skip')
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    text_model = pickle.load(open(Path('Models') / 'text_score.sav', 'rb'))
    t_score = text_model.predict([probas])[0]
    audio_model = pickle.load(open(Path('Models') / 'audio_score.sav', 'rb'))
    a_score = audio_model.predict([emotion_dist])[0]
    score = round((73.755 * a_score + 26.2445 * t_score) / 100, 2)
    
    return render_template('score_analysis.html', a_emo=major_emotion, a_prob=emotion_dist, t_text=text, t_traits=probas, t_trait=trait, t_num_words=num_words, t_common_words=common_words_perso, name=name, position=job_position, score=score)

# Audio Interview
@app.route('/audio', methods=['POST', 'GET'])
def audio_index():
    flash("After pressing the button, you have 15 seconds to answer.")
    return render_template('audio.html', display_button=False, color='#C7EEFF')

@app.route('/audio_recording', methods=['POST', 'GET'])
def audio_recording():
    SER = speechEmotionRecognition()
    rec_sub_dir = Path('tmp') / 'voice_recording.wav'
    
    try:
        SER.voice_recording(str(rec_sub_dir), duration=16)
        flash("Recording over! Evaluate your answer based on emotions expressed.")
    except Exception as e:
        flash(f"Error during recording: {e}")
    
    return render_template('audio.html', display_button=True, color='#00ffad')

@app.route('/audio_analysis', methods=['POST', 'GET'])
def audio_analysis():
    model_sub_dir = Path('Models') / 'audio.hdf5'
    rec_sub_dir = Path('tmp') / 'voice_recording.wav'
    SER = speechEmotionRecognition(str(model_sub_dir))
    
    step = 1
    sample_rate = 16000
    emotions, timestamp = SER.predict_emotion_from_file(str(rec_sub_dir), chunk_step=step * sample_rate)
    
    SER.prediction_to_csv(emotions, Path("static/js/db") / "audio_emotions.txt", mode='w')
    SER.prediction_to_csv(emotions, Path("static/js/db") / "audio_emotions_other.txt", mode='a')
    
    major_emotion = max(set(emotions), key=emotions.count)
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(Path('static/js/db') / 'audio_emotions_dist.txt', sep=',')
    
    df_other = pd.read_csv(Path("static/js/db") / "audio_emotions_other.txt", sep=",")
    major_emotion_other = df_other.EMOTION.mode()[0]
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION == emotion]) / len(df_other)) for emotion in SER._emotion.values()]
    
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(Path('static/js/db') / 'audio_emotions_dist_other.txt', sep=',')
    
    return render_template('audio_analysis.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other)

# Text Interview
@app.route('/text', methods=['POST', 'GET'])
def text():
    return render_template('text.html')

@app.route('/text_analysis', methods=['POST'])
def text_analysis():
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    probas_others, trait_others = save_text_analysis_data(text, probas, traits)
    probas = [int(e * 100) for e in probas]
    
    session['probas'] = probas
    session['text_info'] = {"common_words": [], "num_words": []}
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open(Path("static/js/db") / "words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for line in counts:
            d.write(f"{line},{counts[line]}\n")
    
    with open(Path("static/js/db") / "words_common.txt", "a") as d:
        for line in counts:
            d.write(f"{line},{counts[line]}\n")
    
    df_words_co = pd.read_csv(Path('static/js/db') / 'words_common.txt', sep=',', on_bad_lines='skip')
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric, errors='coerce')
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv(Path('static/js/db') / 'words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    df_words_perso = pd.read_csv(Path('static/js/db') / 'words_perso.txt', sep=',', on_bad_lines='skip')
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    return render_template('text_analysis.html', traits=probas, trait=trait, trait_others=trait_others, probas_others=probas_others, num_words=num_words, common_words=common_words_perso, common_words_others=common_words_others)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/text_input', methods=['POST'])
def text_pdf():
    if 'file' not in request.files:
        flash("No file uploaded.")
        return redirect(request.url)
    
    f = request.files['file']
    if f.filename == '':
        flash("No file selected.")
        return redirect(request.url)
    
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        f.save(file_path)
        
        try:
            text = parser.from_file(str(file_path))['content']
            traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
            probas = get_personality(text)[0].tolist()
            
            probas_others, trait_others = save_text_analysis_data(text, probas, traits)
            probas = [int(e * 100) for e in probas]
            
            session['probas'] = probas
            session['text_info'] = {"common_words": [], "num_words": []}
            
            preprocessed_text = preprocess_text(text)
            common_words, num_words, counts = get_text_info(preprocessed_text)
            
            session['text_info']["common_words"].append(common_words)
            session['text_info']["num_words"].append(num_words)
            
            trait = traits[probas.index(max(probas))]
            
            with open(Path("static/js/db") / "words_perso.txt", "w") as d:
                d.write("WORDS,FREQ\n")
                for line in counts:
                    d.write(f"{line},{counts[line]}\n")
            
            with open(Path("static/js/db") / "words_common.txt", "a") as d:
                for line in counts:
                    d.write(f"{line},{counts[line]}\n")
            
            df_words_co = pd.read_csv(Path('static/js/db') / 'words_common.txt', sep=',', on_bad_lines='skip')
            df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric, errors='coerce')
            df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
            df_words_co.to_csv(Path('static/js/db') / 'words_common.txt', sep=",", index=False)
            common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
            
            df_words_perso = pd.read_csv(Path('static/js/db') / 'words_perso.txt', sep=',', on_bad_lines='skip')
            common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
            
            return render_template('text_dash.html', traits=probas, trait=trait, trait_others=trait_others, probas_others=probas_others, num_words=num_words, common_words=common_words_perso, common_words_others=common_words_others)
        except Exception as e:
            flash(f"Error processing file: {e}")
            return redirect(request.url)
    else:
        flash("Invalid file type. Only PDF files are allowed.")
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)