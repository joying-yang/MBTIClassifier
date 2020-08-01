from flask import Flask, render_template, request
from process import get_mbti

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def index():
    errors = ""
    result = ""
    if request.method == "POST":
        text = str(request.form["text"])
        if len(text) < 100:
            return render_template('index.html').format(result=result, errors="Can you like enter something longer...")
        else:
            result = get_mbti(text)
            return render_template('index.html').format(result=result, errors=errors)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
