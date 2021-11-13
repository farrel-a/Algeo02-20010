from flask import Flask, render_template, request, flash, redirect, url_for
import urllib.request
from pathlib import Path
from src.OperasiMatriks import compressImg
import os
from werkzeug.utils import secure_filename



def create_app():
    app = Flask(__name__)

    UPLOAD_FOLDER = os.getcwd() + "/Website/static/uploads"
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["SECRET_KEY"] = 'Semangat ges'

    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    @app.route("/home")
    def home():
        # Nanti fungsi SVD Bakal ada disini
        return render_template("index.html")

    @app.route("/", methods=["POST", "GET"])
    def save_image():
        if request.method == "POST":
            if 'file' not in request.files:
                flash("No file detected")
                return redirect(url_for('home'))
            file = request.files['file']
            if file.filename == "":
                flash("No image selected")
                return request.url

            if file and allowed_file(file.filename):
                filename = file.filename
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                print(filename)
                file.save(path)
                fn = os.path.splitext(path)
                root,ext = os.path.splitext(path)
                compressed_filename = Path(fn[0] + "_compressed").stem + ext
                compressImg(path, 50)
                print(compressed_filename)
                flash('Image sucsessfully uploaded and displayed')
                return render_template('index.html', filename=filename,com_fn = compressed_filename)

            else:
                flash("Allowed image type are: png,jpg, or jpeg")
                return redirect(request.url)

    @app.route('/display/<filename>')
    def display_image(filename):
        # print('display_image filename: ' + filename)
        return redirect(url_for('static', filename='uploads/' + filename), code=301)

    @app.route('/display/<filename>')
    def display_compressed_image(filename):
        return redirect(url_for('static', filename='uploads/' + filename), code=301)


    return app