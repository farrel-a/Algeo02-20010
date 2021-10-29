from flask import Flask,render_template

def create_app():
    app = Flask(__name__)

    @app.route('/')
    @app.route("/home")
    def home():
        #Nanti fungsi SVD Bakal ada disini
        return render_template("index.html")

    return app