from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import arabic_reshaper
from bidi.algorithm import get_display

# إعداد Flask
app = Flask(__name__)

# نموذج التلخيص
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')  # واجهة لتحميل النصوص

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']  # استلام النص من المستخدم
    # تلخيص النص
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    # تجهيز النص العربي (إن وجد)
    reshaped_summary = arabic_reshaper.reshape(summary)
    display_summary = get_display(reshaped_summary)
    
    return jsonify({"summary": display_summary})

import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
