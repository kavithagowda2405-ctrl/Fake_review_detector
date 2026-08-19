from flask import Flask, render_template, request
from combined_analysis import analyze_review

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "")

        if review.strip():
            result = analyze_review(review)

    return render_template(
        "index.html",
        result=result,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)