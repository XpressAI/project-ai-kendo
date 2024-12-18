import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig

# Add 'evf-sam' submodule path dynamically
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EVF_SAM_DIR = os.path.join(FILE_DIR, "../EVF-SAM")
if EVF_SAM_DIR not in sys.path:
    sys.path.insert(0, EVF_SAM_DIR)

from model.segment_anything.utils.transforms import ResizeLongestSide

def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer on folder")
    parser.add_argument("--version", required=True)
    parser.add_argument("--vis_save_path", default="./infer", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    # Changed from single image to folder input
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output predictions")
    parser.add_argument("--prompt", type=str, default="zebra top left")

    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    '''
    assert img_size == 1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."

    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None

    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook if requested
    if args.precision == "fp16":
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    elif args.precision == "bf16":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    else:
        # fp32 - no autocast
        pass

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Check input folder
    if not os.path.exists(args.input_folder):
        print("Input folder not found: {}".format(args.input_folder))
        sys.exit(1)

    # Prepare output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # initialize model and tokenizer
    tokenizer, model = init_models(args)

    # List all images in the input folder
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [f for f in sorted(os.listdir(args.input_folder)) if f.lower().endswith(valid_extensions)]

    if len(image_files) == 0:
        print("No valid images found in the folder.")
        sys.exit(1)

    prompt = args.prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    for image_name in image_files:
        image_path = os.path.join(args.input_folder, image_name)
        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)
        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        # infer
        pred_mask = model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        # save visualization
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

        # Construct output path
        output_path = os.path.join(args.output_folder, f"{os.path.splitext(image_name)[0]}_vis.png")
        cv2.imwrite(output_path, save_img)
        print(f"Processed {image_path}, saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
