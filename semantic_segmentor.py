from PIL import Image
from lang_sam import LangSAM
import numpy as np

class SemanticSegmentQuery(object):
    """
    语义分割查询接口，基于 LangSAM 模型
    """
    def __init__(self, image_paths, prompts):
        """
        :param image_paths: List of image file paths or a single path string.
        :param prompts: List of text descriptions or a single description string.
        """
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
        else:
            self.image_paths = image_paths
            
        if isinstance(prompts, str):
            self.prompts = [prompts]
        else:
            self.prompts = prompts

class SemanticSegmentResponse(object):
    """
    语义分割响应结果封装
    """
    def __init__(self, results=None):
        """
        :param results: Dictionary containing segmentation results.
                        Format: {"pic_path": {"label": {"scores": ..., "boxes": ..., "masks": ...}, ...}, ...}
        """
        self.results = {}
        if results is not None:
            self.results = results

    def _update(self, pic_path, label, scores, boxes, masks, mask_scores):
        """
        更新单张图片的单个标签结果
        """
        if pic_path not in self.results:
            self.results[pic_path] = {}
        self.results[pic_path][label] = {
            "scores": scores,
            "boxes": boxes,
            "masks": masks,
            "mask_scores": mask_scores
        }
    
    def _dict_results(self):
        return self.results

    def visualize(self, output_dir=None):
        """
        可视化结果
        :param output_dir: 如果提供，将结果保存到该目录；否则直接显示
        """
        from PIL import ImageDraw
        import os

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for pic_path, labels_data in self.results.items():
            try:
                image = Image.open(pic_path).convert("RGBA")
            except Exception as e:
                print(f"Could not open image {pic_path}: {e}")
                continue
            
            # Create a transparent layer for masks
            mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            for label, data in labels_data.items():
                # Generate a random color for each label
                color = tuple(np.random.randint(0, 256, 3).tolist())
                
                boxes = data['boxes']
                masks = data['masks']
                scores = data['scores']

                for i in range(len(boxes)):
                    box = boxes[i]
                    mask = masks[i]
                    score = scores[i]

                    # Draw Box [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw Label
                    text = f"{label}: {score:.2f}"
                    text_pos = (x1, max(0, y1 - 15))
                    draw.text(text_pos, text, fill=color)

                    # Draw Mask
                    if mask.ndim == 2:
                        # Create a solid color image for the mask with transparency (alpha=100)
                        solid_color = Image.new("RGBA", image.size, color + (100,))
                        # Convert mask to PIL Image (L mode)
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                        # Paste the solid color using the mask onto the mask layer
                        mask_layer.paste(solid_color, (0, 0), mask_pil)

            # Composite the image and the mask layer
            result_image = Image.alpha_composite(image, mask_layer)
            result_image = result_image.convert("RGB")

            if output_dir:
                filename = os.path.basename(pic_path)
                save_path = os.path.join(output_dir, f"vis_{filename}")
                result_image.save(save_path)
                print(f"Saved visualization to {save_path}")
            else:
                result_image.show()

class SemanticSegmentorLangSAM(object):
    def __init__(self, options=None):
        self.options = options if options is not None else {}
        
        # 1. 从 options 中提取 sam_type
        sam_type = self.options.get("sam_type", "sam2.1_hiera_small")
        print(f"Loading LangSAM with sam_type: {sam_type}")
        
        # 2. 传递具体的参数给 LangSAM 构造函数
        self.model = LangSAM(sam_type=sam_type)

    def predict(self, query: SemanticSegmentQuery, batch_max_size=1) -> SemanticSegmentResponse:
        """
        Detect objects defined in query.prompts within images at query.image_paths.
        
        Args:
            query (SemanticSegmentQuery): The query object containing image paths and prompts.
            batch_max_size (int): Maximum number of images to process in one batch to avoid OOM.
            
        Returns:
            SemanticSegmentResponse: The response object containing structured results.
        """
        image_paths = query.image_paths
        prompts = query.prompts

        # 3. 从 options 中提取预测阈值
        box_threshold = self.options.get("box_threshold", 0.3)
        text_threshold = self.options.get("text_threshold", 0.25)

        # Prepare the prompt string.
        # If multiple descriptions are given, join them with dots to detect all of them.
        if isinstance(prompts, list):
            # Ensure no trailing dots in individual prompts before joining
            cleaned_prompts = [p.strip().rstrip(".") for p in prompts]
            text_prompt = " . ".join(cleaned_prompts) + "."
        else:
            text_prompt = prompts
            if not text_prompt.endswith("."):
                text_prompt += "."

        ssr = SemanticSegmentResponse()
        total_images = len(image_paths)
        
        # Process in batches
        for start_idx in range(0, total_images, batch_max_size):
            end_idx = min(start_idx + batch_max_size, total_images)
            batch_paths = image_paths[start_idx:end_idx]
            
            print(f"Processing batch {start_idx+1}-{end_idx}/{total_images}...")
            
            # Load images for this batch
            batch_images_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # LangSAM expects a list of prompts matching the list of images
            batch_prompts = [text_prompt] * len(batch_images_pil)

            # Predict
            # When passing lists, LangSAM returns a list of dictionaries
            try:
                # 4. 传递具体的阈值参数给 predict
                raw_results = self.model.predict(
                    batch_images_pil, 
                    batch_prompts, 
                    box_threshold=box_threshold, 
                    text_threshold=text_threshold
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM Error in batch {start_idx+1}-{end_idx}. Try reducing batch_max_size.")
                    import torch
                    torch.cuda.empty_cache()
                raise e
            
            for path, res in zip(batch_paths, raw_results):
                # Fix for 0-d arrays from LangSAM (specifically mask_scores when single detection)
                if "mask_scores" in res and np.ndim(res["mask_scores"]) == 0:
                    res["mask_scores"] = np.array([res["mask_scores"]])
                
                if "scores" in res and np.ndim(res["scores"]) == 0:
                    res["scores"] = np.array([res["scores"]])

                # Get all detected labels for this image
                # 'text_labels' contains the specific prompt matched (e.g. 'vase', 'chair')
                labels = res.get("text_labels", [])
                
                # Identify unique labels to group by
                unique_labels = set(labels)
                
                for label in unique_labels:
                    # Find all indices corresponding to this label
                    indices = [i for i, x in enumerate(labels) if x == label]
                    
                    # Use numpy indexing to extract the subset of data for this label
                    ssr._update(
                        pic_path=path,
                        label=label,
                        scores=res["scores"][indices],
                        boxes=res["boxes"][indices],
                        masks=res["masks"][indices],
                        mask_scores=res["mask_scores"][indices]
                    )
        
        return ssr

class SemanticSegmentor(object):
    """
    语义分割接口类
    """
    def __init__(self, segmentor_type="lang_sam", options=None):
        self.segmentor = None
        if segmentor_type == "lang_sam":
            self.segmentor = SemanticSegmentorLangSAM(options)
        else:
            raise ValueError(f"Unsupported segmentor type: {segmentor_type}")

    def predict(self, query: SemanticSegmentQuery, batch_max_size=8) -> SemanticSegmentResponse:
        return self.segmentor.predict(query, batch_max_size=batch_max_size)

if __name__ == "__main__":
    # Example usage
    options = {
        "sam_type": "sam2.1_hiera_small",
        "box_threshold": 0.3,
        "text_threshold": 0.25
    }
    segmentor = SemanticSegmentor("lang_sam", options)
    
    image_paths = ["asserts/111.png", "asserts/222.png"]  # Example image paths
    prompts = ["vase", "chair"] # Detect both vase and chair
    
    # Create Query object
    query = SemanticSegmentQuery(image_paths, prompts)
    
    # Get Response object
    response = segmentor.predict(query)
    results = response._dict_results()
    
    # Visualize results
    response.visualize()
    
    # Print structure to verify
    for path, labels_data in results.items():
        print(f"Results for {path}:")
        for label, data in labels_data.items():
            print(f"    Label: '{label}' found {len(data['boxes'])} times.")
            print(f"    Scores: {data['scores']}")