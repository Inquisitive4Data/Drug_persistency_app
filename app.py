from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import json
import pickle
from dataset_validation import validate_cols  # Omar's module to validate uploaded dataset
from data_mapping import decode_dataset,encode_dataset

app = Flask(__name__)
with open('DrugLR.pkl', 'rb') as f:
    model = pickle.load(f)

# every upload file will be saved with this name
global template_name
template_name = 'template_dataset.xlsx'
# every upload file will be saved with this name
global uploaded_doc_name
uploaded_doc_name = 'downloaded.xlsx'
# save predicted file as csv and update name
global predicted_file_name_csv
predicted_file_name_csv = 'prediction.xlsx'

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/download-template', methods=['POST'])
def download_csv_template():
    return send_file(template_name,
                     mimetype='text/csv',
                     attachment_filename=template_name,
                     as_attachment=True)

@app.route('/upload-dataset', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(uploaded_doc_name))

        # Validate dataset
        # returns a string object to be displayed to the user. Either success of failure.
        validation_output = validate_cols(uploaded_doc_name)
        # mapping user dataset into numerical dataset
        user_data = pd.read_excel(uploaded_doc_name, engine='openpyxl', index_col=None)
        encoded_dataset = encode_dataset(user_data)
        encoded_dataset.to_excel('encoded.xlsx')
        classification = model.predict(encoded_dataset)
        classification_df= pd.DataFrame()
        classification_df['Persistency_Flag'] = classification
        prediction_df = decode_dataset(classification_df)
        prediction_df.to_excel(predicted_file_name_csv)

        return redirect(url_for('uploaded_successfully', validation_output=validation_output))


@app.route('/uploaded-successfully/<validation_output>')
def uploaded_successfully(validation_output):
    return render_template('redirect_upload_success.html', validation_output=validation_output)

@app.route('/download-prediction', methods=['POST'])
def download_csv_predicted():
    return send_file(predicted_file_name_csv,
                     mimetype='text/csv',
                     attachment_filename=predicted_file_name_csv,
                     as_attachment=True)


if __name__ == '__main__':
    app.run(port=8082, debug=True)
