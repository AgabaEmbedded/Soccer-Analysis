from roboflow import Roboflow
rf = Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")
project = rf.workspace("agabaembedded").project("soccer-torso-number-kzyph")
version = project.version(6)
dataset = version.download("yolov11")
                