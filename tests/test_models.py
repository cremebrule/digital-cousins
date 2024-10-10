import digital_cousins
# If you store the offline dataset elsewhere, please uncomment the following line and put the directory here
# digital_cousins.ASSET_DIR = "~/assets"

import os
from PIL import Image
import numpy as np
import torch
import argparse
from digital_cousins.models.gpt import GPT
import digital_cousins
import omnigibson as og

TEST_DIR = os.path.dirname(__file__)
SAVE_DIR = f"{TEST_DIR}/test_acdc_output"
TEST_IMG_PATH = f"{TEST_DIR}/test_img.png"
CAPTION = "Fridge. Cabinet."

def test_dinov2(args):
    from digital_cousins.models.dino_v2 import DinoV2Encoder
    encoder = DinoV2Encoder()
    img = np.array(Image.open(TEST_IMG_PATH))
    encoder.get_features(img)


def test_gsamv2(args):
    from digital_cousins.models.grounded_sam_v2 import GroundedSAMv2
    gsam = GroundedSAMv2()
    img_source, img = gsam.load_image(TEST_IMG_PATH)
    boxes, logits, phrases = gsam.predict_boxes(img, CAPTION)
    gsam.predict_segmentation(img_source, boxes)


def test_perspective_fields(args):
    from digital_cousins.models.perspective_fields import PerspectiveFields
    PerspectiveFields().estimate_camera_intrinsics(input_path=TEST_IMG_PATH)


def test_depth_anything_2(args):
    from digital_cousins.models.depth_anything_v2 import DepthAnythingV2
    depth_path = f"{TEST_DIR}/test_depth.png"
    DepthAnythingV2().estimate_depth_linear(input_path=TEST_IMG_PATH, output_path=depth_path)


def test_clip(args):
    from digital_cousins.models.clip import CLIPEncoder
    encoder = CLIPEncoder()
    img = np.array(Image.open(TEST_IMG_PATH))
    encoder.get_features(img)

    # Test text too
    phrases = ["bottom cabinet", "table"]
    encoder.get_text_features(phrases)


def test_faiss(args):
    import faiss
    feature_dim = 20
    query_vecs = np.random.random((5, feature_dim))
    dataset_vecs = np.random.random((100, feature_dim))

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(feature_dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(dataset_vecs)
    dists, idxs = gpu_index_flat.search(query_vecs, 3)


def test_gpt(args):
    from digital_cousins.models.gpt import GPT
    gpt = GPT(api_key=args.gpt_api_key, version=args.gpt_version)
    obj_caption_prompt_payload = gpt.payload_get_object_caption(img_path=TEST_IMG_PATH)
    gpt_text_response = gpt(payload=obj_caption_prompt_payload, verbose=True)
    print(f"GPT test response:\n\n{gpt_text_response}")


def test_fm(args):
    import digital_cousins
    from digital_cousins.models.feature_matcher import FeatureMatcher
    from torchvision.ops.boxes import _box_xyxy_to_cxcywh
    from groundingdino.util.inference import load_image
    from digital_cousins.utils.processing_utils import compute_bbox_from_mask

    save_dir = f"{TEST_DIR}/test_fm_outputs"
    obj_mask_fpath = f"{save_dir}/test_img_mask.png"
    topk_models_dir = f"{save_dir}/test_topk_models"
    fm = FeatureMatcher(
        encoder="DinoV2Encoder",
        encoder_kwargs=None,
        verbose=True,
    )

    # Test segmentation
    _, floor_mask_paths = fm.compute_segmentation_mask(
        input_category="floor",
        input_img_fpath=TEST_IMG_PATH,
        save_dir=f"{save_dir}/floor",
    )

    # Test nearest neighbor matching
    image_source, image = load_image(TEST_IMG_PATH)
    boxes, logits, phrases = fm.gsam.predict_boxes(image, "fridge.")
    all_masks = fm.gsam.predict_segmentation(image_source, boxes, multimask_output=True)
    mask = all_masks[0][0]
    Image.fromarray(mask).save(obj_mask_fpath)
    bboxes = _box_xyxy_to_cxcywh(
        torch.tensor(compute_bbox_from_mask(obj_mask_fpath))).unsqueeze(dim=0)

    candidate_imgs_fdirs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category}/snapshot"
                            for og_category in ["fridge", "wine_fridge", "bottom_cabinet", "display_fridge"]]

    model_results = fm.find_nearest_neighbor_candidates(
        input_category="fridge",
        input_img_fpath=TEST_IMG_PATH,
        candidate_imgs_fdirs=candidate_imgs_fdirs,
        candidate_imgs=None,
        candidate_filter=None,
        n_candidates=8,
        save_dir=topk_models_dir,
        visualize_resolution=(640, 480),
        boxes=bboxes,
        logits=logits[0].unsqueeze(dim=0),
        phrases=[phrases[0]],
        obj_masks=mask.reshape(1, 1, *mask.shape),
        save_prefix="fridge",
        remove_background=False,
    )


def test_acdc_step_1(args):
    from digital_cousins.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=True,
        run_step_2=False,
        run_step_3=False,
        step_1_output_path=None,
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline


def test_acdc_step_2(args):
    from digital_cousins.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=True,
        run_step_3=False,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=None,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline

def test_acdc_step_3(args):
    from digital_cousins.pipeline.acdc import ACDC
    pipeline = ACDC()
    pipeline.run(
        input_path=TEST_IMG_PATH,
        run_step_1=False,
        run_step_2=False,
        run_step_3=True,
        step_1_output_path=f"{TEST_DIR}/acdc_output/step_1_output/step_1_output_info.json",
        step_2_output_path=f"{TEST_DIR}/acdc_output/step_2_output/step_2_output_info.json",
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
    )
    del pipeline


# OG test should always be at the end since it requires a full shutdown during termination
def test_og(args):
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    og.launch()

    print()
    print("*" * 30)
    print("All tests successfully completed!")
    print("*" * 30)
    print()
    og.shutdown()


def main(args):
    # Run all tests
    print()
    print("*" * 30)
    print("Starting tests...")
    print("*" * 30)
    print()
    test_dinov2(args)
    test_gsamv2(args)
    test_perspective_fields(args)
    test_depth_anything_2(args)
    test_clip(args)
    test_faiss(args)
    test_gpt(args)
    test_fm(args)
    test_acdc_step_1(args)
    test_acdc_step_2(args)
    test_acdc_step_3(args)
    og.shutdown()

    # Final test -- OG should always come at the end
    # This og test cannot run together with test_acdc_step_3
    # because the simulator can only be launched once, and after calling og.shutdown(), the whole process will terminate
    # test_og(args)


if __name__ == "__main__":
    # Define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_api_key", type=str, required=True,
                        help="GPT API key to use. Must be compatible with GPT model specified")
    parser.add_argument("--gpt_version", type=str, default="4o", choices=list(GPT.VERSIONS.keys()),
                        help=f"GPT model version to use. Valid options: {list(GPT.VERSIONS.keys())}")

    args = parser.parse_args()

    main(args)
