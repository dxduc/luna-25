from glob import glob
import json
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK
import torch
import time

from processor_cpu import MalignancyProcessor as cpu
from processor_cuda import MalignancyProcessor as cuda

INPUT_PATH = Path("uploads/input")
OUTPUT_PATH = Path("uploads/output")
# RESOURCE_PATH = Path("/opt/app/resources")


def transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )


def itk_image_to_numpy_image(input_image):
    """

    Parameters
    ----------
    input_image: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    numpyImage = SimpleITK.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(
        np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header


class NoduleProcessor:
    def __init__(self, check, ct_image_file, nodule_locations, clinical_information, lession_id, series_instance_uid, mode="3D", model_name="videomae"):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        model_name: Name of the model to be used for prediction
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.clinical_information = clinical_information
        self.mode = mode
        self.model_name = model_name
        self.lession_id = lession_id
        self.series_instance_uid = series_instance_uid
        self.start_time = time.perf_counter()
        # self.processor = cpu(
        #     mode=mode, suppress_logs=True, model_name=model_name)
        
        if check == False:
            self.processor = cpu(
                mode=mode, suppress_logs=True, model_name=model_name)
        else:
            self.processor = cuda(
                mode=mode, suppress_logs=True, model_name=model_name)

    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> Dict:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        numpyImage, header = itk_image_to_numpy_image(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(numpyImage, header, [coords[i]])
            malignancy_risk, logits = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)

        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    def load_inputs(self):
        # load image
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        self.annotationIDs = [p["name"]
                              for p in self.nodule_locations["points"]]
        self.coords = np.array([p["point"]
                               for p in self.nodule_locations["points"]])
        # reverse to [z, y, x] format
        self.coords = np.flip(self.coords, axis=1)

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        Returns
        -------
        None
        """
        image, coords, annotationIDs = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(
            annotationIDs), "Number of outputs should match number of inputs"
        # results = {
        #     "name": "Points of interest",
        #     "type": "Multiple points",
        #     "points": [],
        #     "version": {
        #         "major": 1,
        #         "minor": 0
        #     }
        # }

        # Populate the "points" section dynamically
        # coords = np.flip(coords, axis=1)
        # for i in range(len(annotationIDs)):
        #     results["points"].append(
        #         {
        #             "name": annotationIDs[i],
        #             "point": coords[i].tolist(),
        #             "probability": float(output[i])
        #         }
        #     )

        output_data = []
        output_data.append({"SeriesInstanceUID": self.series_instance_uid})
        output_data.append({"LesionID": self.lession_id})
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - self.start_time) * 1000.0
        for i in range(len(annotationIDs)):
            prop = float(output[i])
            output_data.append({
                "probability": prop,
                "predictionLabel": 1 if prop > 0.5 else 0,
                "processingTimeMs": round(processing_time_ms, 2)
            })
        return {"output": output_data}


def run(nodule_locations, clinical_information, chest_ct_file, lession_id, series_instance_uid, mode="3D", model_name="videomae"):
    # Read the inputs
    # input_nodule_locations = load_json_file(nodule_locations_file)
    # input_clinical_information = load_json_file(clinical_information_file)
    input_chest_ct = load_image_path(chest_ct_file)
    # # Read a resource file: the model weights
    # with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
    #     print(f.read())

    # Validate access to GPU
    check = _show_torch_cuda_info()

    # Run your algorithm here
    processor = NoduleProcessor(check, ct_image_file=input_chest_ct,
                                nodule_locations=nodule_locations,
                                clinical_information=clinical_information,
                                lession_id=lession_id,
                                series_instance_uid=series_instance_uid,
                                mode=mode,
                                model_name=model_name)
    malignancy_risks = processor.process()

    # Save your output
    write_json_file(
        location="results/results.json",
        content=malignancy_risks["output"],
    )
    return malignancy_risks["output"] 


def load_json_file(location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_path(location):
    # Đảm bảo location là đối tượng Path để dễ xử lý
    path_obj = Path(location)

    # TRƯỜNG HỢP 1: Nếu location chính là file (Do API truyền vào)
    if path_obj.is_file():
        return str(path_obj)

    # TRƯỜNG HỢP 2: Nếu location là thư mục (Logic cũ)
    # Tìm kiếm các file có đuôi .mha, .tif, .tiff trong thư mục đó
    input_files = (
        glob(str(path_obj / "*.mha"))
    )

    # Kiểm tra an toàn: Nếu không tìm thấy file nào thì báo lỗi rõ ràng
    if not input_files:
        raise FileNotFoundError(f"Không tìm thấy file ảnh (.mha) nào tại: {location}")

    # Lấy file đầu tiên tìm thấy
    result = input_files[0]

    return result


def _show_torch_cuda_info():
    return torch.cuda.is_available()


# if __name__ == "__main__":
#     mode = "3D"
#     model_name = "finetune-hiera"
#     raise SystemExit(run(mode=mode,
#                          model_name=model_name))
