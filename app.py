from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import numpy as np
import base64
from algorithm.logistic_regression import predict_using_logistic_regression
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values from form
            author = request.form["author"]
            article = request.form["article"]
            result = predict_using_logistic_regression(author + " " + article) 
            print(result)
            return render_template("index.html", result=result)

            # # Create a bar chart to display the result
            # fig, ax = plt.subplots()
            # ax.bar(['First Number', 'Second Number', 'Sum'], [num1, num2, result], color=['blue', 'orange', 'green'])
            # ax.set_ylabel('Values')
            # ax.set_title('Sum Visualization')

            # # Save the plot to a BytesIO object
            # img = io.BytesIO()
            # plt.savefig(img, format='png')
            # img.seek(0)  # Rewind the file pointer to the beginning

            # # Encode image as base64
            # img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

            # return render_template("index.html", result=result, img_data=img_base64)
        except ValueError as e:
            print(e)
            result = "Some Error occurred!"
            return render_template("index.html", result=result, img_data=None)

    return render_template("index.html", result=None, img_data=None)

if __name__ == "__main__":
    app.run(debug=True)
