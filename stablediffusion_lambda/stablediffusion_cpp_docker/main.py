import os
import json
import traceback
import subprocess
import time
import uuid
import httpx
import boto3
import random
import re
from PIL import Image
import xbrz
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from fastapi.responses import FileResponse


from fastapi import FastAPI, Header, HTTPException, Request, Form, UploadFile, File
from mangum import Mangum
from botocore.exceptions import NoCredentialsError

#we're going to need a function which takes in the given "path", makes a file list of all the files in that path, and then returns that full file path list.
def list_files(directory):
    full_paths = []  # List to store full paths of files
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            full_paths.append(full_path)
    return full_paths


# Environment Variables
IMAGE_DIMENSION = int(os.environ.get("IMAGE_DIMENSION"))
MODELPATH = list_files(os.environ.get("MODELPATH"))[0]
VAEPATH =list_files(os.environ.get("VAEPATH"))[0]
UPSCALEPATH = list_files(os.environ.get("UPSCALEPATH"))
TAEPATH = os.environ.get("TAEPATH")
SDPATH = os.environ.get("SDPATH")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
stage = os.environ.get('STAGE', None)
openapi_prefix = f"/{stage}" if stage else "/"

app = FastAPI(title="Stable Diffusion on Lambda API")
print("Lambda Function Loaded")



class RNGType(str, Enum):
    std_default = "std_default"
    cuda = "cuda"

class SampleMethod(str, Enum):
    euler_a = "euler_a"
    euler = "euler"
    heun = "heun"
    dpm2 = "dpm2"
    dpm2s_a = "dpm++2s_a"
    dpm2m = "dpm++2m"
    dpm2mv2 = "dpm++2mv2"
    lcm = "lcm"

class Schedule(str, Enum):
    default = "default"
    discrete = "discrete"
    karras = "karras"

class SDMode(str, Enum):
    txt2img = "txt2img"
    img2img = "img2img"
    convert = "convert"

class Txt2ImgDiffuserModel(BaseModel):
    #n_threads: Optional[int] = Field(default=None)
    #mode: SDMode = Field(default=SDMode.txt2img)
    #model_path: str
    #vae_path: str
    #taesd_path: Optional[str] = Field(default=None)
    #esrgan_path: Optional[str] = Field(default=None)
    #controlnet_path: Optional[str] = Field(default=None)
    #embeddings_path: Optional[str] = Field(default=None)
    #lora_model_dir: Optional[str] = Field(default=None)
    #output_path: str = Field(default="output.png")
    #input_path: Optional[str] = Field(default=None)
    #control_image_path: Optional[str] = Field(default=None)
    prompt: str
    negative_prompt: Optional[str] = Field(default=None)
    cfg_scale: float = Field(default=7.0)
    clip_skip: Optional[int] = Field(default=None)
    width: int = Field(default=512)
    height: int = Field(default=512)
    batch_count: int = Field(default=1)
    sample_method: SampleMethod = Field(default=SampleMethod.euler_a)
    #schedule: Schedule = Field(default=Schedule.default)
    sample_steps: int = Field(default=10)
    #strength: float = Field(default=0.75)
    #control_strength: float = Field(default=0.9)
    #rng_type: RNGType = Field(default=RNGType.cuda)
    seed: int = Field(default=-1)
    verbose: bool = Field(default=False)
    vae_tiling: bool = Field(default=False)
    control_net_cpu: bool = Field(default=False)
    canny_preprocess: bool = Field(default=False)
    upscale_repeats: int = Field(default=1)
    use_vae: bool = Field(default=False)
    use_esrgan: bool = Field(default=False)
    use_xbrz: bool = Field(default=False)
    xbrz_scale: int = Field(default=1)
    use_taesd: bool = Field(default=False)




def upscale_image_xbrz(input_path, output_path, scale_factor):
    # Load the image using Pillow
    input_image = Image.open(input_path)
    
    # Convert the image to RGBA (if it's not already in that mode)
    input_image_rgba = input_image.convert('RGBA')
    
    # Upscale the image using xbrz.scale_pillow
    upscaled_image = xbrz.scale_pillow(input_image_rgba, scale_factor)
    
    # Save the upscaled image
    upscaled_image.save(output_path)

def download_from_s3(objectname: str) -> str:
    print("Download from s3")
    s3 = boto3.client('s3')
    local_path = f"/tmp/{objectname.split('/')[-1]}"
    s3.download_file(BUCKET_NAME, objectname, local_path)
    return local_path


def shuffle_string(s: str) -> str:
    print("Shuffle String")
    char_list = list(s)
    random.shuffle(char_list)
    return ''.join(char_list)


def execute_and_return_image(cmd, output,use_xbrz,scale):
    # Create the subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    # Continuously read from stdout
    while True:
        # Read one line from stdout
        line = process.stdout.readline()
        if not line:
            break  # Exit the loop when there is nothing left to read
        print(line, end='')  # Print the line (end='' prevents adding an extra newline)

    # Wait for the subprocess to finish
    process.communicate()  # This will also read any remaining output
    return_code = process.returncode

    # Check the return code
    if return_code != 0:
        # Get any error output
        error_output = process.stderr.read()
        raise HTTPException(status_code=500, detail=error_output)
    if use_xbrz:
        upscale_image_xbrz(output,output,scale)
    # Upload the output
    #return upload_output(output)
    return get_image(output)

async def get_image(image_path: str):
    return FileResponse(image_path)



@app.post("/txt2img")
def execute_txt2img(

    diffuser_model: Txt2ImgDiffuserModel,
    

    ):
    height = IMAGE_DIMENSION
    width = IMAGE_DIMENSION
    print("execute binary")
    output="/tmp/"+shuffle_string(str(uuid.uuid4()).split("-")[-1]+str(time.time()).replace(".",""))+".png"
    print("Now generating command")
    # Construct the command
    cmd = [
        SDPATH,
        "--mode", "txt2img",
        #"--threads", str(diffuser_model.n_threads),
        "--model", str(MODELPATH),
        "--output", str(output),
        "--prompt", str(diffuser_model.prompt),
        "--negative-prompt", str(diffuser_model.negative_prompt),
        "--cfg-scale", str(diffuser_model.cfg_scale),
        #"--strength", str(diffuser_model.strength),
        "--height", str(height),
        "--width", str(width),
        #"--sampling-method", diffuser_model.sample_method,
        "--steps", str(diffuser_model.sample_steps),
        #"--rng", str(diffuser_model.rng_type),
        "--seed", str(diffuser_model.seed),
        #"--schedule", str(diffuser_model.schedule),
        "--upscale-repeats", str(diffuser_model.upscale_repeats)
    ]


    use_vae: bool = Field(default=False)
    use_esrgan: bool = Field(default=False)
    use_xbrz: bool = Field(default=False)
    use_taesd: bool = Field(default=False)
    
    if diffuser_model.verbose:
        cmd.append("--verbose")
    
    #We'll have a separate img2img endpoint
    #if init_img:
    #    cmd.extend(["--init-img", download_from_s3(init_img)])
        
    if diffuser_model.use_vae:
        cmd.extend(["--vae",VAEPATH])
    if diffuser_model.use_esrgan:
        cmd.extend(["--upscale-model",UPSCALEPATH])
    if diffuser_model.use_taesd:
        cmd.extend(["--taesd",TAEPATH])
    print(cmd)
    try:
        return execute_and_return_image(cmd, output,diffuser_model.use_xbrz,diffuser_model.xbrz_scale)
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def upload_output(file_path: str) -> dict:
    print("Upload output")
    s3 = boto3.client('s3')
    file_name = file_path.split("/")[-1]
    s3.upload_file(file_path, BUCKET_NAME, file_name)
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': file_name},
        ExpiresIn=3600
    )
    print(presigned_url)
    return {"presigned_url": presigned_url}



@app.post("/upload")
async def upload_file_to_s3(file: UploadFile = File(...)):
    print("Upload file to s3")
    s3 = boto3.client('s3')
    unique_filename = shuffle_string(str(uuid.uuid4()).replace("-", "") + str(time.time()).replace(".", ""))
    unique_filename += "." + file.filename.split(".")[-1]
    file_content = await file.read()
    s3.put_object(Body=file_content, Bucket=BUCKET_NAME, Key=unique_filename)
    return {"unique_filename": unique_filename}

@app.get("/healthcheck")
def healthcheck():
    print("Healthcheck")
    return {"status": "OK"}

handler=Mangum(app)


