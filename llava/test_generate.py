import sys
from dataclasses import dataclass, field
import torch
from llava.constants import IMAGE_TOKEN_INDEX 
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from abc import abstractproperty
from PIL import Image
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, default="./images/dino.png")
    parse.add_argument("--model_path", type=str, default="./checkpoints/vlora-7b-sft")
    parse.add_argument("--question", type=str, default="Please describe this image.")
    args = parse.parse_args()
    
    
    model_path = args.model_path
    model_name = 'vlora'
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to(torch.float16)
    model = model.cuda()
    
    from PIL import Image
    image_path = args.image_path
    image = Image.open(image_path).convert("RGB")
    args = abstractproperty()
    args.image_aspect_ratio = 'pad'
    image_tensor = process_images([image], image_processor, args).to('cuda',dtype=torch.float16)
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{args.question} ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, images=image_tensor, \
                                    do_sample=True,temperature = 0.7, max_new_tokens=512, use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        print(outputs)