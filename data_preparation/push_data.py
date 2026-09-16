import roboflow

rf = roboflow.Roboflow(api_key="9VCfGRMmzvi61B4c9qu5")
workspace = rf.workspace("agabaembedded")

workspace.upload_dataset(
    r"C:\Users\Agaba_Embedded4\Desktop\Soccer-Tracking\soccer-torso-number-6",                  # path to a structured dataset directory
    "jersey-number-detection-wvznv",                 # project id (created if it doesn't exist)
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0,
    is_prediction=False,           # True for model-generated annotations awaiting review
)