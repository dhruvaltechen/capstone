from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import numpy as np
import base64
from prediction import predict
from accuracies import accuracy_scores, algorithm_labels
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values from form
            algorithm = request.form["algorithm"]
            author = request.form["author"]
            article = request.form["article"]
            result = predict(author + " " + article, algorithm) 
            print(result)
            # return render_template("index.html", result=result)

            # Create a bar chart to display the result
            fig, ax = plt.subplots()
            ax.bar(algorithm_labels, accuracy_scores, color=['blue', 'orange', 'green', 'red', 'purple'])
            ax.set_xlabel('Algorithms')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Visualization')

            # Save the plot to a BytesIO object
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)  # Rewind the file pointer to the beginning

            # Encode image as base64
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

            return render_template("index.html", result=result, img_data=img_base64)
        except ValueError as e:
            print(e)
            result = "Some Error occurred!"
            return render_template("index.html", result=result, img_data=None)

    return render_template("index.html", result=None, img_data=None)

if __name__ == "__main__":
    app.run(debug=True)


# steps to run project:
# 1. pip install -r requirements.txt
# 2. python app.py
