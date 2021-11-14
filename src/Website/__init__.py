from flask import Flask, render_template, request, flash, redirect, url_for
from pathlib import Path
from .OperasiMatriks import compressImg
import os


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
                ratio = int(request.form.get('ratio'))
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                path = os.path.abspath(path)
                print(filename)
                file.save(path)
                fn = os.path.splitext(path)
                root,ext = os.path.splitext(path)
                compressed_filename = Path(fn[0] + "_compressed").stem + ext
                runtime = compressImg(path,ratio)
                print(compressed_filename)
                flash('Image sucsessfully uploaded and displayed')
                return render_template('index.html', filename=filename,com_fn = compressed_filename,rt = runtime,rat=ratio)

            else:
                flash("Allowed image type are: png,jpg, or jpeg")
                return redirect(request.url)

    @app.route('/display/<filename>')
    def display_image(filename):
        # print('display_image filename: ' + filename)
        filename = 'uploads/'+filename
        filename = os.path.abspath(filename)
        return redirect(url_for('static', filename='uploads/' + filename), code=301)

    @app.route('/display/<filename>')
    def display_compressed_image(filename):
        filename = 'uploads/' + filename
        filename = os.path.abspath(filename)
        return redirect(url_for('static', filename='uploads/' + filename), code=301)


    return app