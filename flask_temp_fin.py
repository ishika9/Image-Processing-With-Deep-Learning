
from flask_cors import CORS,cross_origin
from flask import Flask, request, jsonify,Response ,make_response, send_file, send_from_directory
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image
import io
from io import BytesIO
import zipfile
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path


UPLOAD_FOLDER = 'userimages/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
counter = 0

class RequestUtils:
    def __init__(self, req : object) -> None:
        self.req = req
    
    def getFilenameByName(self, name : str) -> str:
        return self.req.files[name].filename
    
    def readFileByName(self, name : str) -> str:
        return self.req.files[name]

    def getReadObjByName(self, name : str) -> str:
        return self.req.files[name]
    
    def getRequestRealIP(self):
        return self.req.headers['X-Real-IP']
    
    def getQueryParam(self, name):
        return self.req.args.get(name)

# ipaddress:port/endpoint/
@app.route("/endpoint/", methods=['GET', 'POST'])
@cross_origin()
def index():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    # print(request.files['zip_file'])
    req = RequestUtils(request)

    if request.method == 'POST':
        try:
            # file = request.files['zip_file']
            file = req.getReadObjByName('zip_file')
            file.save('temp_zipfile.zip')

            images=[]
            # image_path = "C:/Users/Arihant/Desktop/duplicate/data"

            # os.mkdir(image_path)
            
            with zipfile.ZipFile('temp_zipfile.zip', mode='r') as zip_ref:
                zip_ref.extractall('temp')
                path = "C:/Users/Arihant/Desktop/duplicate/temp/101D3200"
                dataset = fo.Dataset.from_images_dir(path)
                # for file in zip_ref.infolist():
                #     print(file.filename)
                #     image = Image.open(file)
                #     image = image.save(f"{image_path}/{file.filename}")

                    # image=Image.open(file)
                    # images.append(image)
            model=foz.load_zoo_model("mobilenet-v2-imagenet-torch")
            embeddings=dataset.compute_embeddings(model)
            # print(embeddings.shape)

            similarity_matrix = cosine_similarity(embeddings)

            n=len(similarity_matrix)
            similarity_matrix= similarity_matrix- np.identity(n)

            id_map = [s.id for s in dataset.select_fields(["id"])]
            #creates a list called id_map that contains the id field for each element in the dataset.
            #The id_map list will contain the id field for each element in the dataset. For example, if the dataset object contains 10 elements, the id_map list will contain 10 elements, one for each element in the dataset
            for idx, sample in enumerate(dataset):
                max_similarity= similarity_matrix[idx].max()
                sample["max_similarity"]= max_similarity
                sample.save()


            dataset.match(F("max_similarity")>0.991)

            thresh = 0.991
            samples_to_remove = set()
            samples_to_keep = set()

            for idx, sample in enumerate(dataset):
                if sample.id not in samples_to_remove:
                    print('Processing sample: ', idx)
                    # Keep the first instance of two duplicates
                    samples_to_keep.add(sample.id)
                    
                    dup_idxs = np.where(similarity_matrix[idx] > thresh)[0]
                    for dup in dup_idxs:
                        # We kept the first instance so remove all other duplicates
                        print('==> duplicates: ', sample.id)
                        samples_to_remove.add(id_map[dup])

                    if len(dup_idxs) > 0:
                        print('=> has duplicates: ', sample.id)
                        sample.tags.append("has_duplicates")
                        sample.save()

                else:
                    sample.tags.append("duplicate")
                    sample.save()
            print(len(samples_to_remove) + len(samples_to_keep))

            dataset.delete_samples(list(samples_to_remove))

            export_dir = "/exp"

            # Export the dataset
            dataset.export(export_dir=export_dir, dataset_type=fo.types.ImageDirectory)
            zipObj = zipfile.ZipFile('output.zip', 'w')
            # zipObj.write(export_dir)

            # for x in os.listdir(export_dir):
            #     if (x.endswith(".JPG")):
            #         zipObj.write(x)

            pics= Path(export_dir).glob('*.JPG')
            for x in pics:
                zipObj.write(x)
            #     print(type(x))

            # close the Zip File
            # zipObj.close()
            send_file(zipObj,
                      mimetype = 'zip',
                      download_name='output.zip', 
                      as_attachment = True)

            zipObj.close()
            
            resp={'yes': 'yes'}

        except Exception as e:
            resp = {'Message': f'Exception, {e}'}
            print(f'exception: {e}')

        # del req
        return jsonify(resp)
        

        # return send_file(export_dir,
        #                  mimetype = 'zip',
        #                  download_name='output.zip',
        #                  as_attachment = True)
    
    return """
        <html><body>
            <h2>Upload file</h2>
            <form action="" method="post" enctype="multipart/form-data">
            enter zip file : <input type="file" name="zip_file" /><br />
            <input type="submit" />
            </form>
        </body></html>
        """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True)
    cors = CORS(app,resources={"/endpoint/": {"origins": "*"}})
    app.config['CORS_HEADERS'] = 'Content-Type'

# C:/Users/Arihant/Desktop/duplicate/