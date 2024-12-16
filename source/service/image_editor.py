from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from fastapi import UploadFile
import requests
from configuration.config import Config, ModelArgs
import io
import logging

logging.basicConfig(level=logging.INFO)

class ImageEditorService:
    def __init__(self):
        """
        A service class for image editing using a conditional diffusion model InstructPix2Pix.

        Attributes:
            unet (UNet2DConditionModel): Instance of UNet model for image conditioning.
            vae (AutoencoderKL): Instance of variational autoencoder model.
            text_encoder (CLIPTextModel): Instance of CLIP text encoder model.
            tokenizer (CLIPTokenizer): Instance of CLIP tokenizer.
            noise_scheduler (DDPMScheduler): Instance of scheduler for diffusion models.
            model (StableDiffusionInstructPix2PixPipeline): Instance of instruction-based image processing pipeline.
    """

        self.unet = UNet2DConditionModel.from_pretrained(Config.UNET_PATH)
        self.vae = AutoencoderKL.from_pretrained(Config.MODEL_PATH, subfolder="vae")
        self.text_encoder = CLIPTextModel.from_pretrained(Config.MODEL_PATH, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(Config.MODEL_PATH, subfolder="tokenizer")
        self.noise_scheduler = DDPMScheduler.from_pretrained(Config.MODEL_PATH, subfolder="scheduler")
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            Config.MODEL_PATH,
            unet=self.unet,
            text_encoder=self.text_encoder,
            vae=self.vae,
        )
    def read_image_from_file(self, file_content: bytes) -> Image.Image:
        """
        Reads an image from bytes content.

        Args:
            file_content (bytes): Byte content of the image file it can be png, jpeg or jpg file.

        Returns:
            Image.Image: PIL Image object.

        Raises:
            ValueError: If the file content not an image.
        """
        logging.info("Reading image from file content")
        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            logging.info("Image read successfully")
            return image
        except Exception as e:
            raise ValueError("Cannot identify image file")
    
    def download_image(self, url: str) -> Image.Image:
        """
        Downloads an image from a given URL.

        Args:
            url (str): URL of the image to download.

        Returns:
            Image.Image: PIL Image object.

        Raises:
            ValueError: If the image cannot be downloaded from the URL.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image = Image.open(response.raw)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            raise ValueError(f"Could not download image from URL: {url}")
        
    def run_inference(self, image: Image.Image, instruction: str) -> Image.Image:
        """
        Runs inference on the input image with the given instruction.

        Args:
            image (Image.Image): Input image for processing.
            instruction (str): Instruction or prompt for image processing.

        Returns:
            str: Output path where the processed image is saved.

        """
        logging.info(f"Processing image with prompt: {instruction}")
        images = self.model(instruction,
                            image=image,
                            num_inference_steps=ModelArgs.num_inference_steps,
                            image_guidance_scale=ModelArgs.image_guidance_scale,
                            guidance_scale=ModelArgs.guidance_scale
                            ).images
        output_image = images[0]
        output_image.save(Config.OUTPUT_PATH)
        return Config.OUTPUT_PATH
    async def process_image(self, instruction: str, file: UploadFile = None, url: str = None) -> str:
        """
        Processes an image based on provided input (file or URL) and instruction.

        Args:
            instruction (str): Instruction or prompt for image processing.
            file (UploadFile, optional): Uploaded file object containing the image. Defaults to None.
            url (str, optional): URL of the image to download. Defaults to None.

        Returns:
            str: Output path where the processed image is saved.

        Raises:
            ValueError: If both file and URL are provided or if neither is provided.
        """
        if (file and url) or (not file and not url):
            raise ValueError("Please provide either an image file or a URL, but not both.")
        if file:
            logging.info("Reading file content")
            file_content = await file.read()
            image = self.read_image_from_file(file_content)
        elif url:
            logging.info(f"URL: {url}")
            image = self.download_image(url)
            
        return self.run_inference(image, instruction)
    