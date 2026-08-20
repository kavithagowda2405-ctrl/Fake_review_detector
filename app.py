from flask import Flask, render_template, request
from combined_analysis import analyze_review
import pandas as pd

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

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    summary = None
    results_table = None
    error =  None

    if request.method == 'POST':
        file = request.files['csv_file']

     # Check 1: no file selected
        if not file or file.filename == '':
            error = "No file selected. Please choose a CSV file to upload."
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        # Check 2: wrong file type
        if not file.filename.lower().endswith('.csv'):
            error = "Invalid file type. Please upload a .csv file."
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            error = f"Could not read the CSV file. Make sure it's a valid, properly formatted CSV. ({str(e)})"
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        # Check 3: empty file
        if df.empty:
            error = "The uploaded CSV is empty. Please upload a file with review data."
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        # Check 4: find a usable text column
        possible_text_cols = ['text', 'review', 'review_text', 'Review']
        text_col = None
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[0]  # fallback to first column

        # Check 5: drop empty/missing rows in the text column
        df = df.dropna(subset=[text_col])
        df = df[df[text_col].astype(str).str.strip() != '']

        if df.empty:
            error = f"No valid review text found in column '{text_col}'. Please check your CSV content."
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        # Limit batch size to avoid long waits/crashes on huge files
        max_rows = 200
        if len(df) > max_rows:
            df = df.head(max_rows)
            error = f"File had more than {max_rows} rows — only the first {max_rows} were analyzed."

        try:
            analyses = df[text_col].apply(lambda x: analyze_review(str(x)))
            df['fake_probability'] = analyses.apply(lambda a: round(a['fake_probability'] * 100, 2))
            df['ai_likelihood_score'] = analyses.apply(lambda a: round(a['ai_likelihood_score'] * 100, 2))
            df['verdict'] = analyses.apply(lambda a: a['verdict'])
        except Exception as e:
            error = f"Something went wrong while analyzing the reviews. ({str(e)})"
            return render_template('batch.html', summary=summary, results=results_table, error=error)

        summary = {
            'total': len(df),
            'likely_fake': int((df['verdict'] != 'Likely Real').sum()),
            'likely_real': int((df['verdict'] == 'Likely Real').sum()),
            'pct_fake': round((df['verdict'] != 'Likely Real').mean() * 100, 1)
        }

        df_display = df[[text_col, 'fake_probability', 'ai_likelihood_score', 'verdict']].rename(
            columns={text_col: 'text'}
        )
        results_table = df_display.to_dict('records')

    return render_template('batch.html', summary=summary, results=results_table)

if __name__ == '__main__':
    app.run(debug=True)