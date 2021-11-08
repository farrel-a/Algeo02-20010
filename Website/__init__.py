from flask import Flask,render_template,request,flash,redirect,url_for
import urllib.request
import  os
from werkzeug.utils import secure_filename



def create_app():
    app = Flask(__name__)

    UPLOAD_FOLDER = 'static/uploads'
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    @app.route("/home")
    def home():
        #Nanti fungsi SVD Bakal ada disini
        return render_template("index.html")

    @app.route("/",methods = ["POST","GET"])
    def display_converted_image():
        if 'file' not in request.files:
            flash("No file detected")
            return redirect(url_for('home'))
        file = request.files['file']
        if file.filename == "":
            flash("No image selected")
            return (request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            flash('Image sucsessfully uploaded and displayed')
            return (render_template('index.html', filename=filename))
        else:
            flash("Allowed image type are: png,jpg, or jpeg")
            return redirect(request.url)


    return app