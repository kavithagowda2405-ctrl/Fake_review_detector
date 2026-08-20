from flask import Flask, render_template, request
from combined_analysis import analyze_review

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        review_text = request.form['review_text']
        analysis = analyze_review(review_text)

        fake_prob = round(analysis['fake_probability'] * 100, 2)
        ai_prob = round(analysis['ai_likelihood_score'] * 100, 2)

        result = {
            'text': review_text,
            'fake_probability': fake_prob,
            'real_probability': round(100 - fake_prob, 2),
            'ai_likelihood_score': ai_prob,
            'human_likelihood_score': round(100 - ai_prob, 2),
            'verdict': analysis['verdict']
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)