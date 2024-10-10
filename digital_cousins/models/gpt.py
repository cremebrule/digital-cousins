import requests
import base64


class GPT:
    """
    Simple interface for interacting with GPT-4O model
    """
    VERSIONS = {
        "4v": "gpt-4-vision-preview",
        "4o": "gpt-4o",
        "4o-mini": "gpt-4o-mini",
    }

    def __init__(
            self,
            api_key,
            version="4o",
    ):
        """
        Args:
            api_key (str): Key to use for querying GPT
            version (str): GPT version to use. Valid options are: {4o, 4o-mini, 4v}
        """
        self.api_key = api_key
        assert version in self.VERSIONS, f"Got invalid GPT version! Valid options are: {self.VERSIONS}, got: {version}"
        self.version = version

    def __call__(self, payload, verbose=False):
        """
        Queries GPT using the desired @prompt

        Args:
            payload (dict): Prompt payload to pass to GPT. This should be formatted properly, see
                https://platform.openai.com/docs/overview for details
            verbose (bool): Whether to be verbose as GPT is being queried

        Returns:
            None or str: Raw outputted GPT response if valid, else None
        """
        if verbose:
            print(f"Querying GPT-{self.version} API...")

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.query_header, json=payload)
        if "choices" not in response.json().keys():
            print(f"Got error while querying GPT-{self.version} API! Response:\n\n{response.json()}")
            return None

        if verbose:
            print(f"Finished querying GPT-{self.version}.")

        return response.json()["choices"][0]["message"]["content"]

    @property
    def query_header(self):
        """
        Returns:
            dict: Relevant header to pass to all GPT queries
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def encode_image(self, image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            str: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def payload_get_object_caption(self, img_path):
        """
        Generates custom prompt payload for object captioning

        Args:
            img_path (str): Absolute path to image to caption

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_image = self.encode_image(img_path)

        prompting_text_system = "You are an expert in image captioning. " + \
                            "### Task Description ###\n\n" + \
                            "The user will give you an image, and please provide a list of all objects ([object1, object2, ...]) visible in the image. " + \
                            "For objects with visible and openable doors and drawers, please also return the number of doors (those with revolute joint, that can rotate around an axis) and drawers (those with prismatic joint, that can slide along a direction).\n\n" + \
                            "### Special Requirements ###\n\n" + \
                            "1. Treat each item as a single entity; avoid breaking down objects into their components. For instance, mention a wardrobe as one object instead of listing its doors and handles as separate items; " + \
                            "mention a pot plant/flower as a whole object instead of listing its vase separately.\n\n" + \
                            "2. When captioning, please do not include walls, floors, windows and any items hung from the ceiling in your answer, but please include objects installed or hung on walls.\n\n" + \
                            "3. When captioning, you can use broader categories. For instance, you can simply specify 'table' instead of 'short coffee table'.\n\n" + \
                            "4. Please caption all objects, even if some objects are closely placed, or an object is on top of another, or some objects are small compared to other objects. " + \
                            "However, don't come up with objects not in the image.\n\n" + \
                            "5. Please do not add 's' or 'es' suffices to countable nouns. For example, you should caption multiple apples as 'apple', not 'apples'.\n\n" + \
                            "6. When counting the number of doors and drawer, pay attention to the following:\n\n" + \
                            "(1). A child link cannot be a door and a drawer at the same time. When you are not sure if a child link is a door or a drawer, choose the most likely one.\n" + \
                            "(2). Please only count openable doors and drawers. Don't include objects with fixed and non-openable drawers/shelves/baskets (e.g., shelves/baskets/statis drawers of bookshelves, shelves, storage carts). For these objects, just give me the caption (e.g., bookshelf, shelf, storage cart).\n\n" + \
                            "Example output1: [banana, cabinet(3 doors & 3 drawers), chair]\n" + \
                            "Example output2: [wardrobe(2 doors), table, storage cart]\n" + \
                            "Example output3: [television, apple, shelf]\n" + \
                            "Example output4: [cabinet(8 drawers), desk, frying pan]\n\n\n"

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        
        content_user = [
            {
                "type": "text",
                "text": "Now please provide a list of all objects ([object1, object2, ...]) visible in the image below.\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        object_caption_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 500
        }

        return object_caption_payload

    def payload_select_object_from_list(self, img_path, obj_list, bbox_img_path, nonproject_obj_img_path):
        """
        Generates custom prompt payload for selecting an object from a list of objects

        Args:
            img_path (str): Absolute path to image to infer object selection from
            obj_list (list of str): List of previously detected objects
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_image = self.encode_image(img_path)
        base64_target_obj_mask_image = self.encode_image(nonproject_obj_img_path)
        base64_target_obj_bbox_image = self.encode_image(bbox_img_path)

        prompting_text_system = "You are an expert in object captioning.\n\n" + \
                            "### Task Overview ###\n" + \
                            "The user will show you a scene, a target object in the scene highlighted by a bounding box, " + \
                            "and the mask of the target object in the scene where only the target object is shown in its original color and all other objects are masked as black pixles.\n\n" + \
                            "Next, the user will give you a list of object captions, each describing one or multiple objects in the scene. " + \
                            "Your task is to select the best caption from this list that most accurately describes the target object.\n\n" + \
                            "### Special Requirements ###\n\n" + \
                            "Please follow these guidelines when selecting your answer:\n\n" + \
                            "1. If multiple captions could be correct, choose the one that most accurately describes the target object.\n\n" + \
                            "2. Select a caption for the entire object. For instance, if the target object is a cabinet with doors, choose 'cabinet' instead of 'door.' " + \
                            "Similarly, if the object is a plant in a jar, choose 'plant' instead of 'jar'.\n\n" + \
                            "3. Focus on the target object.\n\n" + \
                            "(1) There may be occlusions or nearby objects included in the bounding box and the mask. " + \
                            "For example, if a bowl is in front of a cup, the bounding box and mask for the cup might contain part of the bowl (due to occlusion). " + \
                            "If some objects are too close to the target object, like an apple in a plate, then the bounding box and mask for the plate can include part of the apple (due to adjacency).\n" + \
                            "Ensure that you caption the intended object, not the occluding or adjacent one.\n\n" + \
                            "(2) In the object mask, the target object is shown in its original color, while all other objects are masked as black. " + \
                            "If multiple objects are on top or adjacent to the target object, the mask of the target object can contain a outline of other objects. " + \
                            "Please focus on the target object.\n\n" + \
                            "(3) Each bounding box and mask refers to only one target object, and the box usually centers at the target object. " + \
                            "You can use these two principles to help infer the target object under occlusion and adjacency.\n\n" + \
                            "4. If the target object is heavily occluded, you can use your common sense to infer the most likely caption of the target object. " + \
                            "For example, if multiple fruits are in the plate, the plate might be heavily occluded. Suppose the list of object captions contain 'fork' and 'plate', " + \
                            "based on common sense, you can infer the target object is more possible to be a plate, because it is strange that fruits are 'in' or 'on' a fork.\n\n" + \
                            "Similar situations happen when multiple objects are on top of in the target object, causing occusion to the target object. When you give your answer, please make sure it is not counter-intuitive.\n\n" + \
                            "5. Only select a caption from the provided list. Do not create any new caption that is not in the list.\n\n" + \
                            "6. Provide only the most appropriate caption, without any explanation.\n\n" + \
                            'Example output: banana\n\n'

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": "The above image shows a scene. " + \
                        "The following image shows the mask of the target object, where only the target object is shown in its original color with all other objects masked out as black."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_target_obj_mask_image}"
                }
            },
            {
                "type": "text",
                "text": "The following image shows the same object highlighted by a bounding box." 
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_target_obj_bbox_image}"
                }
            },
            {
                "type": "text",
                "text": f"The list of object captions of the scene is: {obj_list}.\n\n." + \
                    "Now please select the best caption from the list for the target object."
            }
        ]

        object_selection_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",  
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }

        return object_selection_payload

    def payload_count_drawer_door(self, caption, bbox_img_path, nonproject_obj_img_path):
        """
        Generates custom prompt payload for selecting an object from a list of objects

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_target_obj_mask_image = self.encode_image(nonproject_obj_img_path)
        base64_target_obj_bbox_image = self.encode_image(bbox_img_path)

        prompting_text_system = f"You are an expert in indoor design, and articulated furniture design.\n" + \
                                f"Your task is to count the number of doors (revolute) and drawers (prismatic) of an object/furniture." + \
                                f"I will give you an image showing a scene in our everyday life where an object in the same scene highlighted by a bounding box, and the mask of the object. " + \
                                f"Please tell me how many doors (those with revolute joint, that can rotate around an axis) and drawers (those with prismatic joint, that can slide along a direction) does the target object have.\n" + \
                                "When counting the number of doors and drawer, pay attention to the following:\n" + \
                                "1. Do not count closely positioned doors/drawers as one single doors/drawers:\n" + \
                                "E.g.(1). Do not regard several doors near each other as a single door. For example, two doors next to each other horizontally with opposite open direction should be defined as two doors, not one door.\n" + \
                                "E.g.(2). Do not regard several drawers stacked vertically or horizontally as a single drawer.\n" + \
                                "In other words, as long as a door or a drawer can be opened independently from other doors or drawers, it should be defined as a separate door or drawer.\n" + \
                                "2. A child link cannot be a door and a drawer at the same time. When you are not sure if a child link is a door or a drawer, choose the most likely one.\n" + \
                                "3. Please give the most appropriate answer without explaination.\n" + \
                                "Example output: (3 doors & 3 drawers)\n" + \
                                "Example output: (2 doors & 1 drawers)\n" + \
                                "Example output: (0 doors & 2 drawers)\n" + \
                                "Example output: (2 doors & 0 drawers)\n\n"

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        object_selection_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The following image shows the target object ({caption}) by a bounding box in a real world scene."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_target_obj_bbox_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"The following image shows the mask of the target object ({caption}), where only the target object is shown in its original color, while all other objects are masked as black. The mask might contain some noise.\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_target_obj_mask_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Based on the instructions I gave you, please tell me the door and drawer count for the target object in the correct format."
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return object_selection_payload

    def payload_nearest_neighbor(
            self,
            img_path,
            caption,
            bbox_img_path,
            candidates_fpaths,
            nonproject_obj_img_path,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)
        mask_obj_img = self.encode_image(nonproject_obj_img_path)

        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will present you an image showing a real-world scene, an object of interest in the scene, and a list of candidate orientations of an asset in a simulator.\n" + \
                        "Your task is to select the most geometrically similar asset as a digital twin of a real world object."

        prompt_user_1 = "### Task Overview ###\n" + \
                f"I will show you an image of a scene. " + \
                f"I will then present you an image of the same scene but with a target object ({caption}) highlighted by a bounding box, " + \
                "and another image showing the mask of the same object.\n" + \
                f"I will then present you a list of candidate assets in my simulator.\n" + \
                f"Your task is to choose the asset with highest geometric similarity to the target object ({caption}) so that I can use the asset to represent the target object in my simulator. " + \
                f"In other words, I want you to choose a digital twin for the target object ({caption}).\n\n" + \
                "### Special Requirements ###\n" + \
                "1. I have full control over these assets (as a whole), which means I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. " + \
                "For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio.\n" + \
                "2. When the target object is partially occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                "3. Also notice that the candidate asset snapshots are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                "4. Consider which asset, after being modified (reoriented, repositioned, rescaled, ratio changed, texture altered, relit), resembles the target object most closely. " + \
                "Geometry (shape) similarity after the aforementioned modifications is much more critical than appearance similarity.\n" + \
                "5. You should consider not only the overall shape, but also key features and affordance of the target object's category. " + \
                "For example, if it is a mug, consider if it has a handle and if some candidate assets have a handle. " + \
                "If they both have handles, which asset has the most similar handle as the target object.\n" + \
                "6. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                "Please return only the index of the most suitable asset's snapshot. Do not include explanations.\n" + \
                "Example output:2\n" + \
                "Example output:6\n" + \
                "Example output:1\n\n\n" + \
                "Now, let's take a deep breath and begin!\n"

        prompt_text_user_final = f"The following are a list of assets you can choose to represent the {caption}. " + \
                        f"Please choose the asset with highest geometric similarity to the target object ({caption}), i.e., choosing the best digital twin for the target object."

        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}"
                }
            },
            {
                "type": "text",
                "text": "The above image shows a scene in the real world. " + \
                        f"The following image shows the same scene with the target object ({caption}) highlighted by a bounding box."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the object is shown in its original color, and black pixels are other objects or background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{mask_obj_img}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]
        
        for i, candidate_fpath in enumerate(candidates_fpaths):
            text_prompt = f"image {i + 1}:\n"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            cand_base64 = self.encode_image(candidate_fpath)
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_base64}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload

    def payload_articulated_nearest_neighbor(
            self,
            img_path,
            caption,
            bbox_img_path,
            candidates_fpaths
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)

        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will present you an image showing a real-world scene, an object of interest in the scene, and a list of candidate orientations of an asset in a simulator.\n" + \
                        "Your task is to select the most geometrically similar asset as a digital twin of a real world object."

        prompt_text_user_1 = "### Task Overview ###\n" + \
                        f"I will show you an image showing a scene in the real world or another simulator, and a {caption} bounded by a bounding box in the scene. " + \
                        f"I will then present you a list of candidate assets in my simulator similar to the {caption}.\n" + \
                        f"Your task is to choose the asset with highest geometric similarity to the target object ({caption}) so that I can use the asset to represent the target object in my simulator. " + \
                        f"In other words, I want you to choose a digital twin for the target object ({caption}).\n\n" + \
                        "### Special Requirements ###\n" + \
                        "I have full control over these assets, which means that I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                        "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio. " + \
                        "When the target object is partly occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                        "Additionally, I cannot split a door or a drawer into two, or merge two doors or drawers into one. Nor can I transform a door into a drawer or vice versa. " + \
                        "Also notice that the assets are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                        f"The {caption} is an articulated object, meaning that it has doors, or drawers, or both. " + \
                        f"Therefore, when selecting the best asset, pay close attention to the following criteria: \n" + \
                        "1. Which asset has similar doors/drawers layout as the target object.\n" + \
                        "2. Handle type of each door/drawer.\n" + \
                        "3. After modification (reorientation, repositioning, rescaling, ratio change, texture alteration, relighting), which asset has the most similar (ideally identical) arrangement of drawers and doors as the target object, in terms of relative size and location. " + \
                        "Geometry similarity after the aforementioned modifications is much more critical than appearance similarity. \n" + \
                        "4. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                        "5. Please return only the index of the most suitable asset's snapshot. Do not include any explanation.\n" + \
                        "Example output:4\n\n\n" + \
                        "Now, let's take a deep breath and begin!\n"
        
        prompt_text_user_2 = "The above image shows a scene in the real world. " + \
                        f"The following image shows the same scene with the target object ({caption}) highlighted by a bounding box."
                           

        prompt_text_user_3 = f"The following are a list of assets you can choose to represent the {caption}." + \
                        f"Please choose the asset with highest geometric similarity to the target object ({caption}), i.e., choosing the best digital twin for the target object. Do not include explanation."

        content = [
            {
                "type": "text",
                "text": prompt_text_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_2
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_3
            }
        ]

        for i, candidate_fpath in enumerate(candidates_fpaths):
            text_prompt = f"image {i + 1}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            cand_base64 = self.encode_image(candidate_fpath)
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_base64}",
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload
    
    def payload_nearest_neighbor_pose(
            self,
            img_path,
            caption,
            bbox_img_path,
            nonproject_obj_img_path,
            candidates_fpaths,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor in terms of
        orientation.

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image
            candidates_fpaths (list of str): List of absolute paths to candidate images

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        mask_obj_img = self.encode_image(nonproject_obj_img_path)

        prompt_text_system = "You are expert in orientation estimation and feature matching.\n\n" + \
                    "### Task Description ###\n" + \
                    f"The user will present you an image showing a real-world scene. " + \
                    "The user will then show an image of the same scene with an object highlighted by a bounding box, and the mask of the same object. " + \
                    "This object is represented in a simulator by a digital twin asset. " + \
                    "The user can re-oriente the asset (rotate the asset around its local z axis), and rescale it to accurately match the target object in the input image.\n\n" + \
                    "The user aims to match the orientation of the asset to the target object in the camera frame (i.e., from the viewer's ponit of view). " + \
                    "The user will present you a list of candidate orientations.\n\n" + \
                    "Your task is to select the best orientation of the asset from the candidate orientations that best match the orientation of the real target object from the viewer's point of view.\n\n" + \
                    "### Special Considerations ###\n" + \
                    "Please keep the following in mind:\n\n" + \
                    "1. There might be other objects in the image. Please focus on the target object.\n\n" + \
                    "2. Please select the best orientation in the camera frame, i.e., the orientations are with respect to the viewer. " + \
                    "For example, if the image is taken from a 45 degree lateral view from the left of the target object's frontal face, " + \
                    "then you should select the orientation where the asset is also observed from a 45 degree lateral view from left of the asset's frontal face.\n" + \
                    "If the target object is angled to the left, then you should select the orientation that the asset is also angled to the left viewing from the camera.\n" + \
                    "If the target object is angled to the right, then you should select the orientation that the asset is also angled to the right viewing from the camera.\n" + \
                    "If the object in the input image faces the camera (does not angled to left or right), then you should select the orientation where the asset faces the camera.\n" + \
                    "So on and so forth.\n\n" + \
                    "3. The candiates may not have a perfect orientation. Please select the nearest one.\n" + \
                    "For example, if the image is taken from a 45 degree lateral view from the left of the target object's frontal face, " + \
                    "but there may be no candidate snapshot taken from a 45 degree lateral view from the left of the corresponding asset. " + \
                    "Suppose there are only frontal view snapshot and snapshot taken from a 45 degree lateral view from the right of the corresponding asset, " + \
                    "please select the frontal view because it has smaller orientation difference with the correct orientation (front to left is smaller than right to left).\n\n" + \
                    "4. When selecting the best orientation, you should first identify common features of the digital twin asset and the target object that can define 'orientation'. " + \
                    "For example, cabinets have the same orientation if their frontal faces are facing the same direction (from the viewer's point of view), where the face with doors and drawers are usually considered as the frontal face of a cabinet. " + \
                    "For other objects, you should also consider key features that define orientation of the category, like the back of a chair, and the handle of a spoon.\n\n" + \
                    "5. Parts of the object may be occluded. Please use partially observable features or common sense to identify key features for determining orientation. " + \
                    "For instance, if the handles of a wardrobe are occluded, you can infer the frontal face by 'the face with the doors/drawers is usually the frontal face'. " + \
                    "If even the doors are occluded, apply common sense, such as 'the back of a wardrobe usually faces the wall, so the opposite face is likely the front,' to determine the frontal face and orientation.\n\n" + \
                    "6. You only need to consider orientation, not rescaling. Keep in mind that the user can rescale the asset along each directions without keeping the relative ratio after you determine the orientation. " + \
                    "For example, a sofa asset may be wide, while the real world sofa may be narrow. " + \
                    "After you determine the orientation, the user can rescale the sofa asset along its local horizontal axis to make it as narrow as the real world sofa. " + \
                    "Thus when you select the best orientation, you should focus on their common features (i.e., the back if they both have a back) even though one of them is wider.\n\n" + \
                    "7. Please return only the index of the most suitable orientation of the digital twin without explanations. The index of candidate orientations start from 0.\n\n" + \
                    "Example output:4"

        content_system = [
            {
                "type": "text",
                "text": prompt_text_system
            }
        ]

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The above image shows a scene in the real world. The following image shows the target object ({caption}) bounded by a bounding box."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the object is shown in its original color, while all other objects and background are masked out as black."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{mask_obj_img}"
                }
            },
            {
                "type": "text",
                "text": "The following images show candidate orientations of the digital twin asset with starting index 0:\n\n"
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"orientation {i}:\n"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
                "type": "text",
                "text": "Please take a deep breath, and now please select the nearest orientation that best matches the target object from the viewer's point of view without explanation. Please strictly follow all instructions.\n"
            })

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return NN_payload
    
    def payload_filter_wall(
            self,
            img_path,
            candidate_fpath,
    ):
        """
        Prompt determining whether a mask is part of a wall/backsplash

        Args:
            img_path (str): Absolute path to image to infer object selection from
            candidate_fpath (str): Absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        cand_img_base64 = self.encode_image(candidate_fpath)

        prompt_text_system = "You are an expert in indoor design and plane fitting.\n\n" + \
                    "The user will provide an image and a mask. Your task is to determine if the mask is a wall or backsplash in the original image."
        
        prompt_text_user = "### Task Description ###\n" + \
                        "I will provide an image showing a scene, and a mask where only the target wall or backsplash is shown in its original color, while all other objects are masked as black.\n" + \
                        "Your task is to distinguish if the mask is part of a wall or backsplash. If yes, return y; If no, return n.\n\n" + \
                        "### Special Requirements ###\n" + \
                        "Pay attention to the following:\n" + \
                        "1. Pixels with their original color are pixels belonging to the mask, while black pixles are outside the mask.\n" + \
                        "2. It is fine if the mask contains pixels belonging to another wall or other objects, but if more than half of the mask contains pixels of other objects, please return n.\n" + \
                        "3. The mask does not need to include a whole wall/backsplash. As long as it includes a part of a wall/backsplash without including a significant part of other objects, please return y.\n\n" + \
                        "### Example Outputs ###\n" + \
                        "Example output (If the mask is part of a wall/backsplash): y\n" + \
                        "Example output (If the mask is not part of a wall/backsplash): n\n\n\n" + \
                        "Now take a deep breath and begin!"

        content_system = [{
            "type": "text",
            "text": prompt_text_system
        }]

        content = [
            {
                "type": "text",
                "text": prompt_text_user
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The above image shows a scene in the real world.\n" + \
                    "The following image shows a possible mask for a wall/backsplash in the image. Please determine if it is part of a wall/backsplash following given instructions."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_img_base64}",
                }
            }
        ]

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return payload
    
    def payload_mount_type(
            self,
            caption,
            bbox_img_path,
            obj_and_wall_mask_path,
            candidates_fpaths
    ):
        """
        Prompt determining whether an object is on the floor or mounted on the wall (and which wall)

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            obj_and_wall_mask_path (str): Absolute path to segmented object and walls image
            candidates_fpaths (list of str): List of absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        bbox_base64 = self.encode_image(bbox_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        obj_and_wall_mask_base64 = self.encode_image(obj_and_wall_mask_path)

        # Surprisingly, putting instructions under system role works better
        prompting_text_system = "You are an expert in indoor design, orientation estimation, and plane fitting.\n\n" + \
                        "### Task Description ###\n" + \
                        "The user will present an image showing a scene from the real world, with an object highlighted by a bounding box. " + \
                        "The user wants to reconstruct this scene in a simulator, so it's crucial to determine whether the object is installed/fixed on wall/backsplash or not.\n" + \
                        "An object is either installed/fixed on wall (e.g., a picture hung on a wall, a kitchen cabinet installed on a backsplash, a cabinet whose back is installed on a wall and side is installed on another wall), or not (e.g., a table put on the floor, a coffee machine on a table).\n" + \
                        "The user will install all objects installed/fixed on wall and disable it from falling down when importing in the simulator; " + \
                        "for objects not installed/fixed on wall, the user will put the object on object/floor beneath it.\n" + \
                        "The user will sequentially show you masks of each wall/backsplash plane in the scene, where a proportion of each wall/backsplash is shown in its original color, and all other elements are masked in black.\n" + \
                        "Note that only a proportion of each wall is included in the corresponding mask. You should regard each wall/backsplash as a whole plane, not only the proportion covered by the mask.\n" + \
                        "Your task is to classify the target object based on the following two options:\n\n" + \
                        "Type 1: If the target object is installed/fixed on a wall/backsplash's plane, return 'wall' followed by the index of the plane(s) the object is installed/fixed on.\n\n" + \
                        "If the object is installed/fixed on multiple walls, return the indices of the walls separated by a comma, e.g., 'wall1,wall3'. " + \
                        "This happens when an object is installed at the corner of two orthogonal wall planes, where the back face of the object is installed on a wall, and a side face is installed on the orthogonal wall.\n\n" + \
                        "Type 2: If the target object is not installed/fixed on the wall/backsplash's plane, meaning it rests on the floor or other objects, such that the object would fall if everything below it were removed, return 'floor'. " + \
                        "Be cautious: an object close to the floor might still be installed/fixed on a wall/backsplash.\n\n" + \
                        "### Special Requirements ###\n" + \
                        "Please keep the following in mind:\n" + \
                        "1. You can use common sense when determining the type. For instance, fridges are rarely installed/fixed on a wall, but usually put on the floor (Type 2) although possibly aligned with a wall, whereas cabinets could fit any of the two types; " + \
                        "Objects placed on top of other objects are rarely installed/fixed on a wall, like objects on tables and countertops.\n" + \
                        "These are heuristics from common sense, but if your observation clearly counter the above statements, please follow the observation.\n" + \
                        "2. For an object with doors and drawers like cabinets, wardrobes and refrigerators, pay attention to see if its back is installed on a wall/backsplash plane. " + \
                        "Sometimes the back of such an object is installed on the plane, but due to occlusion you cannot see where the object masks physical contact with the plane.\n" + \
                        "3. Treat each wall/backsplash as an entire 3D plane. The wall mask may only show part of the plane, and the area an object makes physical contact with a wall (Type 1) may not be visible. For instance, in a kitchen, a backsplash may only be a small part of a larger plane that multiple objects (e.g., cabinets) are installed/fixed on. " + \
                        "Such objects should be classified as Type 1 even if the contact area is not included in the mask, or even far away from the mask.\n" + \
                        "4. You can refer to objects belonging to the same category of the target object and placed closely to the target object in the scene as reference. " + \
                        "Multiple instances of the same object category placed together usually has the same installation type. " + \
                        "For example, multiple cabinets aligned horizontally are either all Type 1 or all Type 2. " + \
                        "A typical example would be all top and bottom cabinets in a kitchen installed/fixed on the same backsplash plane.\n\n" + \
                        "5. Please provide the most appropriate answer without explanation.\n\n" + \
                        "### Example Outputs ###\n" + \
                        "Here are some example outputs corresponding to different mounting types:\n" + \
                        "Example output of Type 1 (Installed/fixed on a single wall): wall1\n" + \
                        "Example output of Type 1 (Installed/fixed on more than one walls): wall2,wall3\n" + \
                        "Example output of Type 2 (Not installed/fixed on a wall): floor"

        content_system = [{
            "type": "text",
            "text": prompting_text_system
        }]
        
        content = [
            {
                "type": "text",
                "text": f"The following image shows the target object ({caption}) bounded by a bounding box in a real world scene."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": "The following images show wall(s) in the image. Please regard walls as planes as mentioned in my instructions."
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"wall{i}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
            "type": "text",
            "text": f"To help you better decide if the target object ({caption}) is installed/fixed on one or multiple wall(s)/backsplash(es), " + \
                    "I also provide the following image shows all wall(s)/backsplash(es) and the target object in their original color, " + \
                    "while all other objects and planes are masked as black."
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{obj_and_wall_mask_base64}",
            }
        })

        content.append({
            "type": "text",
            "text": f"Now please choose the installation type of the target object ({caption}) following instructions."
        })

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return payload

    def payload_align_wall(
            self,
            caption,
            bbox_img_path,
            nonproject_obj_img_path,
            obj_and_wall_mask_path,
            candidates_fpaths
    ):
        """
        Prompt determining whether an object not fixed/installed on a wall is aligned with a wall (and which wall) to help adjust orientation

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image
            obj_and_wall_mask_path (str): Absolute path to segmented object and walls image
            candidates_fpaths (list of str): List of absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        bbox_base64 = self.encode_image(bbox_img_path)
        obj_mask_base64 = self.encode_image(nonproject_obj_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        obj_and_wall_mask_base64 = self.encode_image(obj_and_wall_mask_path)

        # Surprisingly, putting instructions under system role works better
        prompting_text_system = "You are an expert in indoor design, orientation estimation, distance estimation, and plane fitting.\n" + \
                "Your task is to determine if an object in a scene is both aligned with and makes physical contact with one or multiple walls or backsplashes in the scene." + \
                "### Task Description ###\n" + \
                "I will provide a real world image where an object is highlighted by a bounding box, and the caption and mask of the same object. " + \
                "I will then present masks of each wall or backsplash plane in the scene. In these masks, a portion of each wall or backsplash will be shown in its original color, while all other elements are shown in black. " + \
                "It's important to consider each wall/backsplash as an entire 3D plane, not just the portion shown in the mask.\n\n" + \
                "Your task is to determine if the target object highlighted by the bounding box is both aligned with and strictly touches one or multiple walls/backsplashes I shown you:\n\n" + \
                "Case 1: If the target object is both aligned with and touches a wall/backsplash plane, return 'wall' followed by the index of the plane the object makes physical contact with.\n\n" + \
                "If the object is aligned with and makes physical contact with multiple walls, return the indices of the walls separated by a comma, e.g., 'wall1,wall3'. " + \
                "This happens when an object is at the corner of two wall planes, where one face of the object is aligned with and makes physical contact with a wall, and another face is aligned with and makes physical contact with the adjacent wall.\n\n" + \
                "Case 2: Otherwise, return 'floor'.\n" + \
                "In other words, if an object is positioned along a wall(s)/backsplash(es) but does not make physical contact with it, you should return 'floor';\n" + \
                "Or if an object makes physical contact with a wall(s)/backsplash(es), but does not align with it (e.g., only a corner contacts a wall/backsplash), you should return 'floor'.\n\n" + \
                "### Technical Interpretation ###\n" + \
                "'Align with' might be vague in semantics. I will provide other interpretations here:\n" + \
                "'An object is aligned with a wall' means the normal vector of one of the faces of the object is the same as the wall's normal vector.\n" + \
                "Another interpretation would be: If an object is aligned with a wall, then the wall's normal vector must be on the object's local x or y axis.\n\n" + \
                "Also note that I am asking for wall(s)/bachsplash(es) that the target object is both aligned with and make a physical contact with. " + \
                "You should also make sure that the target object makes physical contact with all wall wall(s)/backsplash(es) you returned.\n\n" + \
                "### Special Considerations ###\n" + \
                "Please keep the following in mind:\n\n" + \
                "1. Consider a wall/backsplash as a single but entire 3D plane.\n" + \
                "You should regard a wall/backsplash as a single 3D plane, because there might be multiple 3D planes in the scene. Some of them may be other walls, while some of them might be partitions or screens. " + \
                "Other 3D planes may even be parallel with the wall/backsplash plane. You should only focus on the wall/backsplash covered by the mask.\n" + \
                "You should regard a wall/backsplash as an entire 3D plane because possibly only a portion of a wall/backsplash is visible in the mask. The area where an object makes a physical contact with a wall may not always be visible in the image. " + \
                "For instance, in a kitchen or other dining-related scenes, a backsplash might be a small part of a larger plane that includes multiple objects like cabinets and fridges. These objects may fall into Case 1, even if their contact area with the wall is not directly visible.\n\n" + \
                "2. It is possible that an object is aligned a wall plane but does not make a physical contact with it. Then you should not return that wall. " + \
                "Similarly, it is possible that an object is aligned with multiple wall planes, but only makes physical contact with one or two of them. Then you should only return walls that the object makes physical contact with.\n\n" + \
                "3. Here are useful criteria to determine if an object makes physical contact with a wall:\n" + \
                "(1) If there are other object(s) between the target object and the wall, then the object is impossible to make a physical contact with the wall, so you should not return that wall;\n" + \
                "(2) If you can see or infer the target object has distance with the wall, the object does not touch the wall, so you should not return that wall.\n" + \
                "4. Sometimes due to occlusion, the back of the target object may not be visible. " + \
                "When there is a wall/backsplash behind the target object, you should make inference based on the image and common sense if the back face makes a physical contact with the wall/backsplash behind it. " + \
                "This is common for objects with doors and drawers, like wardrobes, refrigerators, and cabinets. If they align with and make physical contact with one or multiple walls, " + \
                "you usually cannot see if they do touch the wall behind it. In this case, if you infer that the object is highly likely to make a physical contact with the wall behind it, or the distance is very small, you can return that wall.\n\n" + \
                "5. Pay attention to relatively large furnitures, like sofa, refrigerators, cabinets, wardrobes, and so on. They are more offen aligned with and make physical contact with wall(s)/backsplash(es). " + \
                "But this is my own experience. Please put your observation as higher priority.\n\n" + \
                "6. Only objects with a clear concept of 'faces' (e.g., 'frontal face', 'back') and the overall shape is a cuboid can be Case 1. For example, cabinets, fridges, and microwave ovens have a clear concept of faces, where the face with doors and drawers are usually considered as the frontal face. " + \
                "And their overall shape is roughly a cuboid. " + \
                "Objects without a clear definition of faces and objects whose overall shape is not a cuboid, such as bowls, cups, or flowers, should not be classified as aligned with and touching wall plane(s) (Case 1), as resizing them to fit a wall is impractical and unrealistic. (Imagine resizing a cup to fit the wall behind it could result in an unsymmetric cup that is long in the direction to the wall). " + \
                "Notice that I am not saying objects with 'faces' must be Case 1. For those objects, you should see if it is aligned with and makes a physical contact with the wall.\n\n" + \
                "7. For small moveble objects like cups, apples, and bags of chips, you should be very careful if you want to classify them as Case 1, because resizing them along the direction pointing toward the wall could lead to larger size in that direction. Since they are small in size, " + \
                "large changes of size in one direction could lead to unrealistic shape.\n\n" + \
                "8. The index of wall/backsplash planes will start from 0. Please provide the most appropriate answer without explanation.\n\n" + \
                "### Example Outputs ###\n" + \
                "Example output: wall1\n" + \
                "Example output: wall1,wall3\n" + \
                "Example output: floor"

        content_system = [{
            "type": "text",
            "text": prompting_text_system
        }]

        content = [
            {
                "type": "text",
                "text": f"The following image shows the target object ({caption}) bounded by a bounding box in a real world scene. Please focus on the target object ({caption})."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the target object is shown in its original color, and black pixels are other objects or background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{obj_mask_base64}",
                }
            },
            {
                "type": "text",
                "text": "The following images show all wall(s) in the image. Please regard walls as planes as mentioned in my instructions."
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"wall{i}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
            "type": "text",
            "text": f"To help you better decide if the target object ({caption}) is aligned with and makes a physical contact with one or multiple wall(s)/backsplash(es), " + \
                    "I also provide the following image shows all wall(s)/backsplash(es) and the target object in their original color, " + \
                    "while all other objects and planes are masked as black."
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{obj_and_wall_mask_base64}",
            }
        })

        content.append({
            "type": "text",
            "text": f"Now please respond following instructions."
        })

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return payload
    
    @classmethod
    def extract_captions(cls, gpt_text):
        """
        Extracts captions from GPT text. Assumes text is a list.
        During prompting, we prompt GPT to give object captions in a list, so we can localize objects by localizing '[', ']' and ','

        Args:
            gpt_text (str): Raw GPT response, which assumes captions are included as part of a list

        Returns:
            list of str: Extracted captions
        """
        # Remove leading and trailing whitespaces and the surrounding brackets
        cleaned_str = gpt_text.strip().strip('[]')

        # Split the string by comma and strip extra spaces from each object
        raw_objects_list = [obj.strip() for obj in cleaned_str.split(',')]

        # Remove redundant quotes
        cleaned_strings = [str.strip("'").strip('"').lower() for str in raw_objects_list]
        objects_list = [f"{obj_name}" for obj_name in cleaned_strings]  # Enclose each string in double quotes

        return objects_list
