"""
Image file adapter with OCR and vision model support.
Supports Tesseract OCR and OpenAI Vision API.
"""

import io
import base64
import logging
from typing import List, Dict, BinaryIO, Optional, Any, Union, Literal
from enum import Enum
from PIL import Image, ImageFile, ImageFilter, ImageOps
import pytesseract
from pydantic import BaseModel, Field
from . import ContentAdapter, ContentProcessingError
import openai

# Configure logging
logger = logging.getLogger(__name__)

class ExtractionMode(str, Enum):
    TEXT_ONLY = "text_only"
    MEANING_ONLY = "meaning_only"
    BOTH = "both"

class VisionModelType(str, Enum):
    TESSERACT = "tesseract"
    OPENAI = "openai"
    PADDLEOCR = "paddleocr"

class VisionModelConfig(BaseModel):
    """Configuration for vision models.
    
    For local models, you can use any OpenAI-compatible API endpoint by setting:
    - `api_base`: URL of your local model server (e.g., "http://localhost:5000/v1")
    - `api_key`: Can be empty or any string for local models
    - `model_name`: The model name your local API expects
    """
    model_type: VisionModelType = Field(
        default=VisionModelType.TESSERACT,
        description="Type of vision model to use"
    )
    model_name: str = Field(
        default="gpt-4-vision-preview",
        description="Model name/identifier (for OpenAI or local model)"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the API (for local models or custom endpoints)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (leave empty for local models)"
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.2,
        description="Sampling temperature (0-2)"
    )
    extraction_mode: ExtractionMode = Field(
        default=ExtractionMode.TEXT_ONLY,
        description="What to extract from the image: text, meaning, or both"
    )
    meaning_prompt: str = Field(
        default="Describe the key elements and overall meaning of this image in detail. Include any text, objects, and their relationships.",
        description="Prompt to use for extracting meaning from the image"
    )
    text_prompt: str = Field(
        default="Extract all text from this image exactly as it appears.",
        description="Prompt to use for text extraction"
    )
    timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for API requests"
    )

class ImageAdapter(ContentAdapter):
    """Adapter for image files with OCR and vision model support."""
    
    # Supported MIME types for images
    SUPPORTED_TYPES = [
        'image/jpeg',
        'image/png',
        'image/tiff',
        'image/bmp',
        'image/gif',
        'image/webp',
        'image/x-portable-pixmap',
        'image/x-portable-graymap',
        'image/x-portable-bitmap'
    ]
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        model_config: Optional[Union[Dict[str, Any], VisionModelConfig]] = None
    ):
        """
        Initialize the ImageAdapter.
        
        Args:
            tesseract_path: Path to Tesseract executable (if not in PATH)
            model_config: Configuration for the vision model. Can be:
                - A dictionary that will be converted to VisionModelConfig
                - A VisionModelConfig instance
                - None to use defaults
        """
        super().__init__()
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Initialize model configuration
        if model_config is None:
            self.model_config = VisionModelConfig()
        elif isinstance(model_config, dict):
            self.model_config = VisionModelConfig(**model_config)
        else:
            self.model_config = model_config
        
        # Set OpenAI client configuration
        self.openai_api_key = getattr(self.model_config, 'api_key', None)
        self.openai_base_url = getattr(self.model_config, 'api_base', None)
        
        # Lazy load OpenAI client when needed
        self._openai_client = None
        # Cache last OCR meta for metadata reporting
        self._last_ocr_meta: Dict[str, Any] = {}
        # Cache last math meta for metadata reporting
        self._last_math_meta: Dict[str, Any] = {}
        # Cache pix2tex model instance
        self._pix2tex_model = None

    # -------- Preprocessing & Confidence Utilities --------
    def _preprocess_image(self, image: Image.Image, deskew: bool = True, denoise: bool = True, rotate: bool = True, lang: Optional[str] = None) -> Image.Image:
        """Basic preprocessing: EXIF auto-orient, deskew via Tesseract OSD, optional denoise."""
        try:
            # Correct orientation using EXIF tags if present
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        processed = image.convert('L')  # grayscale can improve OCR stability

        # Light denoise
        if denoise:
            try:
                processed = processed.filter(ImageFilter.MedianFilter(size=3))
            except Exception:
                pass

        # Deskew/rotation using Tesseract OSD
        if deskew or rotate:
            try:
                # pytesseract image_to_osd works on image or bytes
                osd = pytesseract.image_to_osd(processed, lang=lang or 'eng')
                # Parse angle from OSD output e.g., "Rotate: 270\nOrientation in degrees: 270\n"
                angle = 0
                for line in osd.splitlines():
                    if 'Orientation in degrees' in line or line.startswith('Rotate:'):
                        try:
                            angle = int(''.join(ch for ch in line if ch.isdigit())) % 360
                            break
                        except Exception:
                            continue
                if angle and angle != 0:
                    # PIL rotates counter-clockwise for positive angles if expand=True
                    processed = processed.rotate(360 - angle, expand=True, resample=Image.BICUBIC)
            except Exception:
                # If OSD not available or fails, skip rotation
                pass

        return processed

    def _tesseract_confidence(self, image: Image.Image, **kwargs) -> float:
        """Compute mean confidence from Tesseract data output in [0,1]."""
        try:
            from pytesseract import Output
            data = pytesseract.image_to_data(image, output_type=Output.DICT, **kwargs)
            confs = [float(c) for c in data.get('conf', []) if c not in ("-1", -1, None)]
            if not confs:
                return 0.0
            mean_conf = sum(confs) / len(confs)
            return max(0.0, min(1.0, mean_conf / 100.0))
        except Exception:
            return 0.0

    def _looks_non_latin(self, text: str) -> bool:
        """Heuristic: high proportion of non-ASCII characters suggests non-Latin."""
        if not text:
            return False
        total = len(text)
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return (non_ascii / max(1, total)) > 0.3

    def _detect_math(self, text_hint: str = "", *, image: Optional[Image.Image] = None) -> bool:
        """Lightweight math detector using token density and simple cues."""
        tokens = set("=+-×*/^√∫∑≤≥()[]{}|\\%<>")
        if text_hint:
            count = sum(1 for ch in text_hint if ch in tokens)
            if len(text_hint) >= 5 and (count / len(text_hint)) >= 0.1:
                return True
        # If no text hint, optionally inspect image for long horizontal line (fraction bar)
        if image:
            try:
                # Use PIL to inspect image for long horizontal line
                width, height = image.size
                for y in range(height):
                    if all(image.getpixel((x, y))[0] < 128 for x in range(width)):
                        return True
            except Exception:
                pass
        # Keep it simple to avoid heavy deps; skip if not provided
        return False

    async def _extract_with_pix2tex(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Extract LaTeX using pix2tex (LaTeX-OCR). Returns {'equation_latex': str, 'confidence': float}."""
        try:
            from pix2tex.cli import LatexOCR  # type: ignore
        except Exception as e:
            raise ContentProcessingError(
                "pix2tex (LaTeX-OCR) not installed. Install with 'pip install pix2tex[api]' to enable math extraction."
            ) from e

        try:
            # Keep original as RGB; let pix2tex handle internal preprocessing.
            base_img = image.convert('RGB')
            # Light cleanup only: small white padding, autocontrast, gentle denoise
            try:
                base_img = ImageOps.expand(base_img, border=8, fill='white')
                base_img = ImageOps.autocontrast(base_img, cutoff=2)
                base_img = base_img.filter(ImageFilter.MedianFilter(size=3))
            except Exception:
                pass
            # Optional safety downscale for huge screenshots to avoid extreme sizes
            w0, h0 = base_img.size
            max_h0 = 1024
            if h0 > max_h0:
                new_w0 = max(64, int(w0 * (max_h0 / float(h0))))
                base_img = base_img.resize((new_w0, max_h0))

            # Lazy init the model once
            if self._pix2tex_model is None:
                self._pix2tex_model = LatexOCR()

            # Multi-scale heights to resemble training scales; pick best by heuristic
            heights = kwargs.get('heights', [96, 128, 160, 192, 224, 256])
            candidates = []

            def score_latex(s: str) -> float:
                s = s or ""
                if not s:
                    return 0.0
                valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\\{}^_+-=*/()., <>|[]:;~\\'`\"$")
                valid_ratio = sum(1 for ch in s if ch in valid_chars) / max(1, len(s))
                penalty = 0.0
                if s.count('{') != s.count('}'):
                    penalty += 0.15
                if any(bt in s for bt in ['\u2191', '\u2193', '↑', '↓']):
                    penalty += 0.25
                if s.endswith('='):
                    penalty += 0.1
                return max(0.0, min(1.0, 0.3 + 0.7 * valid_ratio - penalty))

            for hh in heights:
                bw, bh = base_img.size
                if bh != hh:
                    ww = max(32, int(bw * (hh / float(max(1, bh)))))
                    test_img = base_img.resize((ww, hh))
                else:
                    test_img = base_img
                try:
                    eq = self._pix2tex_model(test_img)
                    eq = eq.strip() if isinstance(eq, str) else ""
                except Exception:
                    eq = ""
                sc = score_latex(eq)
                candidates.append((sc, eq, hh))

            best = max(candidates, key=lambda t: t[0]) if candidates else (0.0, "", None)
            conf, eq_best, h_best = best
            return {"equation_latex": eq_best, "confidence": conf, "chosen_height": h_best}
        except Exception as e:
            raise ContentProcessingError(f"pix2tex failed: {str(e)}")
    
    @classmethod
    def supported_mime_types(cls) -> List[str]:
        return cls.SUPPORTED_TYPES
    
    @property
    def openai_client(self):
        """Lazy load OpenAI client with support for local and cloud endpoints."""
        if self._openai_client is None:
            try:
                import openai
                
                client_kwargs = {}
                
                # Always set base_url if provided
                if self.openai_base_url:
                    client_kwargs["base_url"] = self.openai_base_url
                
                # Check if we're using a local endpoint
                is_local = self.openai_base_url and ("localhost" in self.openai_base_url or "127.0.0.1" in self.openai_base_url)
                
                # Always provide the API key, but use a dummy one for local endpoints if not provided
                if not self.openai_api_key and is_local:
                    client_kwargs["api_key"] = "dummy-key"
                elif self.openai_api_key:
                    client_kwargs["api_key"] = self.openai_api_key
                
                # Initialize the client
                self._openai_client = openai.AsyncOpenAI(**client_kwargs)
                
            except ImportError as e:
                raise ImportError(
                    "Please install the openai package to use vision models: "
                    "pip install openai>=1.0.0"
                ) from e
                
        return self._openai_client

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def _extract_with_tesseract(self, image: Image.Image, **kwargs) -> str:
        """Extract text using Tesseract OCR."""
        try:
            return pytesseract.image_to_string(image, **kwargs)
        except Exception as e:
            raise ContentProcessingError(f"Tesseract OCR failed: {str(e)}")

    async def _extract_with_paddleocr(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Extract text using PaddleOCR if available. Returns {'text': str, 'confidence': float}."""
        try:
            # Lazy import to avoid hard dependency
            from paddleocr import PaddleOCR
        except Exception as e:
            raise ContentProcessingError("PaddleOCR not installed. Install with 'pip install paddleocr' to enable fallback.") from e

        lang = kwargs.get('lang', 'en')
        try:
            # Map common language codes; PaddleOCR uses 'en', 'ch', etc.
            ocr = PaddleOCR(use_angle_cls=True, lang=lang if len(lang) <= 2 else 'en', show_log=False)
            # Convert to RGB numpy array
            import numpy as np
            np_img = np.array(image.convert('RGB'))
            result = ocr.ocr(np_img, cls=True)
            lines = []
            scores = []
            for page in result or []:
                for item in page or []:
                    if isinstance(item, list) and len(item) >= 2:
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            txt = text_info[0] or ''
                            sc = text_info[1] or 0.0
                            if txt:
                                lines.append(txt)
                                scores.append(float(sc))
            text = '\n'.join(lines)
            conf = 0.0 if not scores else max(0.0, min(1.0, sum(scores) / len(scores)))
            return {"text": text, "confidence": conf}
        except Exception as e:
            raise ContentProcessingError(f"PaddleOCR failed: {str(e)}")

    async def _extract_with_openai(self, image: Image.Image, prompt: str = None, **kwargs) -> str:
        """Extract text using JoyCaption API."""
        import aiohttp
        import json
        
        base64_image = self._image_to_base64(image)
        
        # Prepare the request payload matching the JoyCaption API format
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "model": self.model_config.model_name,
            "temperature": kwargs.get("temperature", getattr(self.model_config, "temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", getattr(self.model_config, "max_tokens", 512)),
            "top_k": kwargs.get("top_k", getattr(self.model_config, "top_k", 100)),
            "top_p": kwargs.get("top_p", getattr(self.model_config, "top_p", 1.0)),
            "min_p": kwargs.get("min_p", getattr(self.model_config, "min_p", 0.0)),
            "typical_p": kwargs.get("typical_p", getattr(self.model_config, "typical_p", 1.0)),
            "stream": False
        }
        
        try:
            # Make request to the JoyCaption API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.openai_base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", 30))
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"].strip()
                            if content:
                                return content
                            raise ContentProcessingError("Empty content in API response")
                        raise ContentProcessingError("No choices in API response")
                    error_text = await response.text()
                    raise ContentProcessingError(
                        f"API request failed with status {response.status}: {error_text}"
                    )
        except aiohttp.ClientError as e:
            raise ContentProcessingError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise ContentProcessingError(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            raise ContentProcessingError(f"Vision API request failed: {str(e)}")

    async def extract_text(self, file: BinaryIO, **kwargs) -> Dict[str, str]:
        """
        Extract text and/or meaning from an image based on configuration.
        
        Args:
            file: Binary file-like object containing the image
            **kwargs: Additional arguments that may include:
                - prompt: Override the default prompt
                - extraction_mode: Override the default extraction mode
                - Other model-specific parameters
                
        Returns:
            Dictionary with 'text' and/or 'meaning' keys based on extraction_mode
        """
        try:
            # Reset file pointer to start
            file.seek(0)
            
            # Get extraction mode from kwargs or use default from config
            extraction_mode = kwargs.pop('extraction_mode', self.model_config.extraction_mode)
            
            # If only text is needed and we're using Tesseract/PaddleOCR, run OCR chain
            if extraction_mode == ExtractionMode.TEXT_ONLY and self.model_config.model_type in [VisionModelType.TESSERACT, VisionModelType.PADDLEOCR]:
                # Load image once
                image = Image.open(file)
                lang = kwargs.get('lang', 'eng')
                enable_fallback = kwargs.pop('enable_fallback', True)
                threshold = float(kwargs.pop('confidence_threshold', 0.55))
                do_preprocess = kwargs.pop('preprocess', True)
                math_mode = bool(kwargs.pop('math_mode', False))

                if do_preprocess:
                    image_proc = self._preprocess_image(image, deskew=True, denoise=True, rotate=True, lang=lang)
                else:
                    image_proc = image

                # Run Tesseract first
                text_tess = await self._extract_with_tesseract(image_proc, lang=lang)
                conf_tess = self._tesseract_confidence(image_proc, lang=lang)

                chosen_text = text_tess
                chosen_conf = conf_tess
                chosen_engine = 'tesseract'

                # Decide on fallback to PaddleOCR
                needs_fallback = enable_fallback and (conf_tess < threshold or self._looks_non_latin(text_tess))
                if needs_fallback:
                    try:
                        po = await self._extract_with_paddleocr(image_proc, lang=lang)
                        if po.get('text') and po.get('confidence', 0.0) >= chosen_conf:
                            chosen_text = po['text']
                            chosen_conf = float(po.get('confidence', 0.0))
                            chosen_engine = 'paddleocr'
                    except Exception as e:
                        logger.warning(f"PaddleOCR fallback not used: {e}")

                # Optional math route via pix2tex (LaTeX-OCR)
                run_math = math_mode or self._detect_math(chosen_text, image=image_proc)
                if run_math:
                    try:
                        pm = await self._extract_with_pix2tex(image_proc, device=kwargs.get('device', 'cpu'))
                        self._last_math_meta = {
                            'math_detected': True,
                            'equation_latex': pm.get('equation_latex', ''),
                            'math_confidence': float(pm.get('confidence', 0.0)),
                            'equation_engine': 'pix2tex'
                        }
                    except Exception as e:
                        logger.warning(f"pix2tex not used: {e}")

                # Cache meta for metadata extraction
                self._last_ocr_meta = {
                    'ocr_engine': chosen_engine,
                    'confidence_score': float(chosen_conf)
                }

                result_dict = {"text": chosen_text}
                if self._last_math_meta.get('equation_latex'):
                    result_dict["equation_latex"] = self._last_math_meta['equation_latex']
                return result_dict
                
            # For other cases, use the appropriate model
            result = {}
            image = Image.open(file)
            
            # Handle text extraction if needed
            if extraction_mode in [ExtractionMode.TEXT_ONLY, ExtractionMode.BOTH]:
                if self.model_config.model_type == VisionModelType.OPENAI:
                    prompt = kwargs.pop('prompt', self.model_config.text_prompt)
                    result["text"] = await self._extract_with_openai(image, prompt=prompt, **kwargs)
                else:
                    result["text"] = await self._extract_with_tesseract(image, **kwargs)
            
            # Handle meaning extraction if needed
            if extraction_mode in [ExtractionMode.MEANING_ONLY, ExtractionMode.BOTH]:
                if self.model_config.model_type != VisionModelType.OPENAI:
                    logger.warning("Meaning extraction is only supported with OpenAI models. "
                                "Falling back to text extraction.")
                    if "text" not in result:
                        result["text"] = await self._extract_with_tesseract(image, **kwargs)
                else:
                    prompt = kwargs.pop('prompt', self.model_config.meaning_prompt)
                    result["meaning"] = await self._extract_with_openai(
                        image, 
                        prompt=prompt,
                        **kwargs
                    )
            
            return result
                
        except Exception as e:
            raise ContentProcessingError(f"Failed to extract text from image: {str(e)}")
    
    async def extract_metadata(self, file: BinaryIO) -> Dict[str, Any]:
        """
        Extract metadata from an image.
        
        Args:
            file: Binary file-like object containing the image
            
        Returns:
            Dictionary containing image metadata
        """
        try:
            file.seek(0)
            with Image.open(file) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'info': img.info
                }
        except Exception as e:
            raise ContentProcessingError(f"Failed to extract image metadata: {str(e)}")
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """
        Check if the file is a valid image.
        
        Args:
            file: Binary file-like object to check
            
        Returns:
            True if the file is a valid image, False otherwise
        """
        try:
            file.seek(0)
            with Image.open(file) as img:
                # Try to load the first frame to verify it's a valid image
                img.verify()
                return True
        except Exception:
            return False
            raise ContentProcessingError(f"Error extracting text from image: {str(e)}")
    
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract metadata from the image file."""
        try:
            file.seek(0)
            
            # Load image with PIL
            try:
                image = Image.open(file)
                
                # Get basic image properties
                width, height = image.size
                format_ = image.format
                mode = image.mode
                
                # Get EXIF data if available
                exif = {}
                if hasattr(image, '_getexif') and image._getexif():
                    from PIL.ExifTags import TAGS, GPSTAGS
                    for tag, value in image._getexif().items():
                        tag_name = TAGS.get(tag, tag)
                        exif[tag_name] = value
                
                # Get a small preview (first 100 characters of text if any)
                preview_text = ""
                try:
                    text_dict = await self.extract_text(io.BytesIO(file.getvalue()))
                    text_val = text_dict.get('text') if isinstance(text_dict, dict) else str(text_dict)
                    preview_text = (text_val or '')[:100]
                except Exception as e:
                    logger.warning(f"Could not extract preview text: {str(e)}")
                
                meta = {
                    'width': width,
                    'height': height,
                    'format': format_,
                    'mode': mode,
                    'exif': exif,
                    'preview': preview_text,
                    'size_bytes': len(file.getvalue())
                }
                # Attach OCR meta if available
                if self._last_ocr_meta:
                    meta.update(self._last_ocr_meta)
                # Attach math meta if available
                if self._last_math_meta:
                    meta.update(self._last_math_meta)
                return meta
                
            except Exception as e:
                logger.error(f"Error reading image metadata: {str(e)}")
                return {
                    'error': f"Could not read image metadata: {str(e)}",
                    'size_bytes': len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                }
                
        except Exception as e:
            raise ContentProcessingError(f"Error extracting image metadata: {str(e)}")
        finally:
            file.seek(0)
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is a valid image."""
        try:
            file.seek(0)
            image = self._load_image(file)
            return image is not None
        except Exception:
            return False
        finally:
            file.seek(0)
    
    def _load_image(self, file: BinaryIO) -> Optional[Image.Image]:
        """
        Load an image from a file-like object with format detection.
        
        Args:
            file: File-like object containing image data
            
        Returns:
            PIL.Image.Image if successful, None otherwise
        """
        try:
            # Reset file pointer
            file.seek(0)
            
            # Enable loading of truncated images
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            # Try to open the image
            image = Image.open(file)
            
            # Check if it's a valid image
            image.verify()
            
            # Reopen the image after verification
            file.seek(0)
            return Image.open(file)
            
        except Exception as e:
            logger.debug(f"Failed to load image: {str(e)}")
            return None
