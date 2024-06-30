This repository hosts the python scripts to aid in processing the documents of the Lawbot project for labeling and analysis using Azure Document Intelligence

Setup:
<!--Virtual environment guide https://www.youtube.com/watch?v=Y21OR1OPC9A-->
Create a python virtual environment
Activate the virtual environment

run pip install requirements.txt to install dependencies

Download zip file from *Place zipfile URI here*

Create a workspace by calling LawBotDocument.prep_workspace_from_zip method in main.py

Guide:

The LawbotDocument class has methods setting up a workspace and creating LawBotDocument instances of all documents in that workspace as class methods.

LawBotDocument.prep_workspace_from_zip(path_of_zip_file)
Creates the workspace from documents contained in zip a file.
Workspace is created in the same location as where main.py is ran.

LawBotDocument.load_from_workspace()
Will return a list of LawbotDocument instances for all documents found in a workspace

A LawBotDocument instance has methods to analyze a document, create page images for labeling, generating a cleaned document using bounding boxes from labeling and more.

To use the Azure Document Intelligence methods, you will need both the endpoint and key for the azure resource.
When you have the endpoint and key, add them to your environment variables as LAWBOT_DI_KEY and LAWBOT_DI_ENDPOINT .

Some methods of a LawBotDocument instance can operate on either the original document or the cleaned version
Use the DocumentVersion enumeration class to identify the the document to operate on.


