import os
import pydicom
import numpy as np

# extract image from dicom series 
def extract_instances(input_filepath, output_dir):
    # Read the original DICOM file
    ds = pydicom.dcmread(input_filepath)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each image instance
    for i, image in enumerate(ds.pixel_array):
        # Create a new DICOM dataset
        new_ds = pydicom.Dataset()
        
        # Set patient-related attributes
        new_ds.PatientName = ds.PatientName
        new_ds.PatientID = ds.PatientID
        new_ds.PatientSex = ds.PatientSex
        new_ds.PatientBirthDate = ds.PatientBirthDate
        
        # Set study-related attributes
        new_ds.StudyInstanceUID = ds.StudyInstanceUIDa
        
        # Set series-related attributes
        new_ds.SeriesInstanceUID = f"{ds.SeriesInstanceUID}_instance_{i}"
        new_ds.Modality = ds.Modality
        
        # Set image-related attributes
        new_ds.SOPInstanceUID = f"{ds.SeriesInstanceUID}_instance_{i}"
        new_ds.Rows = image.shape[0]
        new_ds.Columns = image.shape[1]
        new_ds.BitsAllocated = 16
        new_ds.SamplesPerPixel = 1
        new_ds.PhotometricInterpretation = "MONOCHROME2"
        new_ds.PixelRepresentation = 0  # unsigned integer
        new_ds.PixelData = image.tobytes()
        
        # Set the appropriate attributes for saving
        new_ds.is_little_endian = True
        new_ds.is_implicit_VR = False
        new_ds.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Save the new DICOM file
        output_filepath = os.path.join(output_dir, f"instance_{i}.dcm")
        pydicom.dcmwrite(output_filepath, new_ds)

# Example usage
input_filepath = "sample.dcm"
output_dir = "output/instances"
extract_instances(input_filepath, output_dir)
