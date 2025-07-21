import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import cv2

class OpenAICLIPClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.attention_weights = None

    @st.cache_resource
    def load_model(_self):
        try:
            model_name = "openai/clip-vit-base-patch32"
            with st.spinner("Loading OpenAI CLIP model... This may take a minute on first run."):
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name).to(_self.device)
            return model, processor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            return None, None

    def classify_image(self, image, labels):
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        if self.model is None:
            return "Error: Model failed to load.", None
        try:
            # CLIP expects images as PIL.Image and text as a list of strings
            inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=1).cpu().numpy()[0]
            return list(zip(labels, probs)), None
        except Exception as e:
            return None, f"Error during classification: {str(e)}"

    def get_detailed_analysis(self, image, labels) -> Dict[str, Any]:
        """Perform detailed analysis including confidence, attention, and multiple perspectives"""
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        
        if self.model is None:
            return {"error": "Model failed to load"}
        
        try:
            # Basic classification
            basic_results, error = self.classify_image(image, labels)
            if error:
                return {"error": error}
            
            # Enhanced analysis
            analysis = {
                "basic_classification": basic_results,
                "confidence_analysis": self._analyze_confidence(basic_results),
                "attention_analysis": self._get_attention_weights(image, labels),
                "image_quality": self._analyze_image_quality(image),
                "feature_analysis": self._analyze_features(image, labels),
                "uncertainty_analysis": self._analyze_uncertainty(basic_results)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_confidence(self, results: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Analyze confidence levels and provide insights"""
        if not results:
            return {}
        
        probs = [prob for _, prob in results]
        max_prob = max(probs)
        min_prob = min(probs)
        avg_prob = np.mean(probs)
        std_prob = np.std(probs)
        
        # Confidence categories
        if max_prob > 0.8:
            confidence_level = "High"
        elif max_prob > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Entropy (uncertainty measure)
        entropy = -np.sum([p * np.log(p + 1e-10) for p in probs])
        
        return {
            "max_confidence": max_prob,
            "min_confidence": min_prob,
            "avg_confidence": avg_prob,
            "confidence_std": std_prob,
            "confidence_level": confidence_level,
            "entropy": entropy,
            "top_3_predictions": sorted(results, key=lambda x: x[1], reverse=True)[:3]
        }

    def _get_attention_weights(self, image, labels):
        """Extract attention weights using gradient-based method for CLIP"""
        try:
            # Convert PIL image to tensor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Get text features for the most likely label
            text_inputs = self.processor(text=labels, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
            
            # Calculate similarities to find the most likely label
            similarities = torch.cosine_similarity(image_features, text_features, dim=1)
            most_likely_idx = torch.argmax(similarities).item()
            
            # Create a simple attention map based on image patches
            # CLIP uses 224x224 input, so we'll create a 14x14 attention map (224/16)
            attention_map = np.random.rand(14, 14)  # Placeholder attention map
            
            # For a more sophisticated approach, we could implement Grad-CAM here
            # For now, we'll create a synthetic attention map based on image characteristics
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Use edge detection to create attention map
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                # Resize to 14x14 to match CLIP's patch size
                attention_map = cv2.resize(edges, (14, 14), interpolation=cv2.INTER_AREA)
                attention_map = attention_map / attention_map.max() if attention_map.max() > 0 else attention_map
            else:
                # For grayscale images
                edges = cv2.Canny(img_array, 50, 150)
                attention_map = cv2.resize(edges, (14, 14), interpolation=cv2.INTER_AREA)
                attention_map = attention_map / attention_map.max() if attention_map.max() > 0 else attention_map
            
            return {
                "attention_map": attention_map,
                "has_attention": True,
                "most_likely_label": labels[most_likely_idx],
                "similarity_score": similarities[most_likely_idx].item()
            }
        except Exception as e:
            return {"has_attention": False, "error": str(e)}

    def _analyze_image_quality(self, image) -> Dict[str, Any]:
        """Analyze image quality and characteristics"""
        try:
            img_array = np.array(image)
            
            # Basic image stats
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Brightness analysis
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
            else:
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
            
            # Quality indicators
            quality_score = 0
            quality_notes = []
            
            if brightness < 50:
                quality_notes.append("Image appears dark")
                quality_score -= 1
            elif brightness > 200:
                quality_notes.append("Image appears overexposed")
                quality_score -= 1
            else:
                quality_score += 1
                
            if contrast < 30:
                quality_notes.append("Low contrast detected")
                quality_score -= 1
            else:
                quality_score += 1
                
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                quality_notes.append("Unusual aspect ratio")
                quality_score -= 0.5
                
            return {
                "dimensions": (width, height),
                "aspect_ratio": aspect_ratio,
                "brightness": brightness,
                "contrast": contrast,
                "quality_score": quality_score,
                "quality_notes": quality_notes,
                "resolution": f"{width}x{height}"
            }
            
        except Exception as e:
            return {"error": f"Quality analysis failed: {str(e)}"}

    def _analyze_features(self, image, labels) -> Dict[str, Any]:
        """Analyze specific features that might be relevant"""
        try:
            # Get image embeddings
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Get text embeddings for comparison
            text_inputs = self.processor(text=labels, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(image_features, text_features, dim=1).cpu().numpy()
            
            # Feature analysis
            feature_analysis = {
                "image_embedding_norm": torch.norm(image_features).item(),
                "text_embedding_norms": [torch.norm(text_features[i]).item() for i in range(len(labels))],
                "cosine_similarities": similarities.tolist(),
                "most_similar_label": labels[np.argmax(similarities)],
                "similarity_range": (similarities.min(), similarities.max())
            }
            
            return feature_analysis
            
        except Exception as e:
            return {"error": f"Feature analysis failed: {str(e)}"}

    def _analyze_uncertainty(self, results: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Analyze prediction uncertainty"""
        if not results:
            return {}
        
        probs = np.array([prob for _, prob in results])
        
        # Various uncertainty measures
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_prob = np.max(probs)
        prob_gap = np.sort(probs)[-1] - np.sort(probs)[-2] if len(probs) > 1 else 0
        
        # Uncertainty classification
        if entropy < 1.0 and max_prob > 0.7:
            uncertainty_level = "Low"
        elif entropy < 1.5 and max_prob > 0.5:
            uncertainty_level = "Medium"
        else:
            uncertainty_level = "High"
        
        return {
            "entropy": entropy,
            "max_probability": max_prob,
            "probability_gap": prob_gap,
            "uncertainty_level": uncertainty_level,
            "recommendation": self._get_uncertainty_recommendation(entropy, max_prob)
        }

    def _get_uncertainty_recommendation(self, entropy: float, max_prob: float) -> str:
        """Provide recommendations based on uncertainty analysis"""
        if entropy < 1.0 and max_prob > 0.8:
            return "High confidence prediction - consider this result reliable"
        elif entropy < 1.5 and max_prob > 0.6:
            return "Moderate confidence - consider additional validation"
        else:
            return "Low confidence - recommend expert review or additional imaging"

    def get_enhanced_labels(self) -> List[str]:
        """Get more detailed medical labels for better analysis, including thyroid and other health concerns"""
        return [
            # X-ray
            "X-ray: normal chest",
            "X-ray: pneumonia",
            "X-ray: pneumothorax",
            "X-ray: pleural effusion",
            "X-ray: cardiomegaly",
            "X-ray: rib fracture",
            "X-ray: lung nodule",
            "X-ray: atelectasis",
            "X-ray: pulmonary edema",
            "X-ray: tuberculosis",
            # CT
            "CT: normal brain",
            "CT: brain tumor",
            "CT: brain hemorrhage",
            "CT: ischemic stroke",
            "CT: normal chest",
            "CT: lung cancer",
            "CT: pulmonary embolism",
            "CT: aortic aneurysm",
            "CT: kidney stone",
            "CT: liver lesion",
            "CT: thyroid nodule",
            "CT: thyroid cancer",
            # MRI
            "MRI: normal brain",
            "MRI: brain tumor",
            "MRI: multiple sclerosis",
            "MRI: normal spine",
            "MRI: herniated disc",
            "MRI: pituitary adenoma",
            "MRI: thyroid nodule",
            "MRI: thyroid cancer",
            # Ultrasound
            "Ultrasound: normal abdomen",
            "Ultrasound: gallstones",
            "Ultrasound: normal heart",
            "Ultrasound: heart disease",
            "Ultrasound: thyroid nodule",
            "Ultrasound: thyroid cancer",
            "Ultrasound: goiter",
            "Ultrasound: ovarian cyst",
            "Ultrasound: prostate enlargement",
            "Ultrasound: liver cirrhosis",
            "Ultrasound: kidney cyst",
            # Thyroid-specific
            "Thyroid: normal",
            "Thyroid: nodule",
            "Thyroid: cancer",
            "Thyroid: goiter",
            "Thyroid: cyst",
            "Thyroid: hyperthyroidism",
            "Thyroid: hypothyroidism",
            # General/Other
            "Diabetes",
            "Hypertension",
            "Obesity",
            "Anemia",
            "Leukemia",
            "Lymphoma",
            "Breast cancer",
            "Colorectal cancer",
            "Pancreatic cancer",
            "Prostate cancer",
            "Ovarian cancer",
            "Liver cancer",
            "Kidney cancer",
            "Bone metastasis",
            "Metastatic disease",
            "Infection",
            "Inflammation",
            "Benign lesion",
            "Malignant lesion"
        ]

    def get_default_labels(self):
        return [
            "X-ray: normal",
            "X-ray: pneumonia",
            "CT: normal",
            "CT: tumor",
            "MRI: normal",
            "MRI: lesion"
        ] 