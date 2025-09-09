import os
import json
import time
import argparse
import numpy as np
from PIL import Image
import pytesseract

# --- MODELS ---
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# Try FAISS, fall back to NumPy
try:
    import faiss  # type: ignore
    USE_FAISS = True
except Exception:
    USE_FAISS = False

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_JSON = "index.json"          # metadata (ocr text + caption)
EMB_TEXT_NPY = "emb_text.npy"     # sentence-transformers embeddings (OCR+caption)
EMB_IMG_NPY = "emb_img.npy"       # CLIP image embeddings
N_TOP = 5

# -----------------------------
# Utilities
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [d], b: [N, d] -> returns [N]
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm

def standardize(v: np.ndarray) -> np.ndarray:
    # z-score to roughly balance scales between spaces
    mu = v.mean()
    sd = v.std() + 1e-9
    return (v - mu) / sd

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# -----------------------------
# Indexer
# -----------------------------
class Indexer:
    def __init__(self):
        # Models
        print(f"[Info] Using device: {DEVICE}")
        print("[Load] BLIP captioning model…")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

        print("[Load] CLIP (image+text) model…")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

        print("[Load] Sentence-Transformers (text) model…")
        self.st = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

        self.records = []             # [{filename, path, ocr, caption}]
        self.emb_text = None          # np.ndarray [N, d_t]
        self.emb_img = None           # np.ndarray [N, d_i]

        # Search backends
        self.text_index = None
        self.img_index = None

    # --- extraction ---
    def ocr_text(self, img: Image.Image) -> str:
        return pytesseract.image_to_string(img)

    def caption_image(self, img: Image.Image) -> str:
        inputs = self.blip_processor(images=img, return_tensors="pt").to(DEVICE)
        out = self.blip.generate(**inputs, max_new_tokens=25)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    # --- embeddings ---
    def embed_text_batch(self, texts):
        # Sentence-Transformers returns torch or numpy; enforce numpy
        embs = self.st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)

    def embed_images_clip(self, pil_images):
        # returns normalized np array
        with torch.no_grad():
            inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)
            img_emb = self.clip.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().numpy().astype(np.float32)

    def build(self, folder: str):
        t0 = time.time()
        paths = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not paths:
            raise SystemExit(f"No images found in {folder}")

        print(f"[Index] Found {len(paths)} images. Extracting OCR+captions…")
        for p in sorted(paths):
            img = load_image_rgb(p)
            ocr = self.ocr_text(img).strip()
            caption = self.caption_image(img).strip()
            self.records.append({
                "filename": os.path.basename(p),
                "path": p,
                "ocr": ocr,
                "caption": caption
            })
            print(f"  ✓ {os.path.basename(p)} | caption: {caption[:60]}")

        # Save metadata
        with open(DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

        # Text embeddings (OCR + caption concatenated)
        print("[Embed] Computing text embeddings (OCR+caption)…")
        text_units = [(rec["ocr"] + " " + rec["caption"]).strip() for rec in self.records]
        self.emb_text = self.embed_text_batch(text_units)
        np.save(EMB_TEXT_NPY, self.emb_text)

        # Image embeddings (CLIP)
        print("[Embed] Computing image embeddings (CLIP)…")
        pil_imgs = [load_image_rgb(rec["path"]) for rec in self.records]
        self.emb_img = self.embed_images_clip(pil_imgs)
        np.save(EMB_IMG_NPY, self.emb_img)

        # Build ANN indices
        self._build_indices()

        print(f"[Done] Indexed {len(self.records)} images in {time.time()-t0:.1f}s.")

    def _build_indices(self):
        if USE_FAISS:
            print("[Index] Building FAISS indices…")
            d_text = self.emb_text.shape[1]
            d_img = self.emb_img.shape[1]
            self.text_index = faiss.IndexFlatIP(d_text)
            self.img_index = faiss.IndexFlatIP(d_img)
            self.text_index.add(self.emb_text)
            self.img_index.add(self.emb_img)
        else:
            print("[Index] FAISS not available. Using NumPy cosine search.")

    # --- searching ---
    def search(self, query: str, top_k: int = N_TOP, weight_text: float = 0.5, weight_img: float = 0.5):
        """
        weight_text: weight for OCR+caption semantic similarity
        weight_img : weight for CLIP image-space similarity (via CLIP's text encoder)
        """
        if not self.records:
            with open(DATA_JSON, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            self.emb_text = np.load(EMB_TEXT_NPY)
            self.emb_img = np.load(EMB_IMG_NPY)
            self._build_indices()

        # Encode query for both spaces
        q_text = self.st.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        with torch.no_grad():
            clip_inputs = self.clip_processor(text=[query], return_tensors="pt").to(DEVICE)
            q_img_t = self.clip.get_text_features(**clip_inputs)
            q_img_t = q_img_t / q_img_t.norm(dim=-1, keepdim=True)
        q_img = q_img_t.cpu().numpy()[0].astype(np.float32)

        # Scores
        if USE_FAISS:
            # Inner product equals cosine since vectors are normalized
            D_text, I_text = self.text_index.search(q_text.reshape(1, -1), len(self.records))
            D_img, I_img = self.img_index.search(q_img.reshape(1, -1), len(self.records))
            s_text = np.zeros(len(self.records), dtype=np.float32)
            s_img = np.zeros(len(self.records), dtype=np.float32)
            s_text[I_text[0]] = D_text[0]
            s_img[I_img[0]]  = D_img[0]
        else:
            s_text = cosine_sim(q_text, self.emb_text)
            s_img  = cosine_sim(q_img,  self.emb_img)

        # Normalize scores and combine
        s_text = standardize(s_text)
        s_img  = standardize(s_img)
        final = weight_text * s_text + weight_img * s_img

        # Top-K
        idxs = np.argsort(-final)[:top_k]
        results = []
        for rank, i in enumerate(idxs, start=1):
            rec = self.records[i]
            results.append({
                "rank": rank,
                "filename": rec["filename"],
                "path": rec["path"],
                "caption": rec["caption"],
                "confidence": float(final[i])  # relative combined score
            })
        return results

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Visual Memory Search (OCR + Visual)")
    parser.add_argument("--folder", type=str, default="screenshots", help="Folder with screenshots")
    parser.add_argument("--build", action="store_true", help="Build (extract+embed+index)")
    parser.add_argument("--query", type=str, default=None, help='Search query, e.g., "error message about auth"')
    parser.add_argument("--wtext", type=float, default=0.5, help="Weight for text space (OCR+caption)")
    parser.add_argument("--wimg", type=float, default=0.5, help="Weight for image space (CLIP text->image)")
    parser.add_argument("--topk", type=int, default=5, help="Top K results")
    args = parser.parse_args()

    idx = Indexer()
    if args.build:
        idx.build(args.folder)

    if args.query:
        results = idx.search(args.query, top_k=args.topk, weight_text=args.wtext, weight_img=args.wimg)
        print("\nTop matches:")
        for r in results:
            print(f"[{r['rank']}] {r['filename']}  | conf={r['confidence']:.3f}")
            print(f"     {r['caption']}")
            print(f"     {r['path']}")
    elif not args.build:
        print("Nothing to do. Use --build to index and/or --query to search.")

if __name__ == "__main__":
    main()
