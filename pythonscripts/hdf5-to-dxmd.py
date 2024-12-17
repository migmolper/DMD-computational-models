import h5py
import os
import xml.etree.ElementTree as ET
from datetime import datetime
import argparse

def create_xdmd(hdf5_filepath):
    # Extract basic file information
    file_name = os.path.basename(hdf5_filepath)
    file_size = os.path.getsize(hdf5_filepath)
    creation_date = datetime.fromtimestamp(os.path.getctime(hdf5_filepath)).isoformat()
    
    # Open the HDF5 file to extract dataset information
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        datasets = []
        def extract_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "Name": name,
                    "Shape": "x".join(map(str, obj.shape)),
                    "DataType": obj.dtype.name,
                    "Description": f"Dataset named {name} with shape {obj.shape} and type {obj.dtype.name}"
                })
        
        hdf5_file.visititems(extract_datasets)
    
    # Create XML structure
    root = ET.Element("HDF5Wrapper")
    
    file_info = ET.SubElement(root, "FileInfo")
    ET.SubElement(file_info, "FileName").text = file_name
    ET.SubElement(file_info, "FileSize").text = str(file_size)
    ET.SubElement(file_info, "CreationDate").text = creation_date
    
    for dataset in datasets:
        dataset_element = ET.SubElement(root, "Dataset")
        ET.SubElement(dataset_element, "Name").text = dataset["Name"]
        ET.SubElement(dataset_element, "Shape").text = dataset["Shape"]
        ET.SubElement(dataset_element, "DataType").text = dataset["DataType"]
        ET.SubElement(dataset_element, "Description").text = dataset["Description"]
    
    # Additional attributes can be added here
    attributes = ET.SubElement(root, "Attributes")
    ET.SubElement(attributes, "Attribute", name="Author").text = "Unknown"
    ET.SubElement(attributes, "Attribute", name="Project").text = "HDF5 Data Analysis"
    ET.SubElement(attributes, "Attribute", name="Software").text = "Custom Script"
    
    # Convert to a pretty XML string
    tree = ET.ElementTree(root)
    xdmd_filename = hdf5_filepath.replace(".hdf5", ".xdmd")
    tree.write(xdmd_filename, encoding="utf-8", xml_declaration=True)
    
    print(f"{xdmd_filename} created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an XDMD wrapper for an HDF5 file.")
    parser.add_argument("hdf5_filepath", type=str, help="Path to the HDF5 file")
    
    args = parser.parse_args()
    create_xdmd(args.hdf5_filepath)
