"""
generate_dataset.py
--------------------
Builds a paired IMAGE + TEXT dataset that simulates real e-commerce product
listings (a product thumbnail image + a product title/description), for the
task of Multimodal Product Category Classification.

WHY A GENERATED DATASET?
This environment has no network access to dataset hosts (Kaggle / Hugging
Face / Amazon Review Data mirrors, etc.), so a programmatically generated,
paired image-text corpus is used instead. It is built to *mimic the
statistical structure* of a real product catalogue:
  - each category has a distinctive but noisy visual template (color/shape/
    rotation jitter), just like real product photography varies,
  - each category has a distinctive but noisy text template (brand, size,
    material, marketing adjectives), just like real listing titles vary,
  - a controlled fraction of samples have an AMBIGUOUS image (heavy visual
    noise) or an AMBIGUOUS text (keyword stripped out), so that neither
    modality alone is sufficient for every sample -- this is what makes the
    fusion task meaningful (see Need Analysis in the report).

The exact same pipeline (Sections: Preprocessing -> Feature Extraction ->
Fusion -> Classification in src/train.py) will run unmodified on a real
paired catalogue (e.g. Amazon Product Metadata + images, Flipkart Product
Images Dataset) that exposes an `image_path,text,label` CSV -- only this
generation script would be swapped for a real data loader.
"""
import os
import random
import csv
import math
from PIL import Image, ImageDraw

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

IMG_SIZE = 64
N_PER_CLASS = 160  # 5 classes -> 800 total samples

CATEGORIES = ["Electronics", "Footwear", "Furniture", "Fashion", "Books"]

# NOTE: brand names are shared across ALL categories on purpose (just like
# real-world multi-category retailers/marketplaces use one umbrella brand
# name across product lines). This prevents brand name from leaking the
# category label through text alone.
BRANDS = ["Zenith", "Vortek", "Nubex", "Corelex", "Pixion", "Strydo",
          "Fleetfoot", "Woodline", "Havenly", "Threado", "Fabrico",
          "PenCraft", "Bindwell", "Urbanest", "Corex"]

GENERIC_PHRASES = [
    "great value for money",
    "top rated by customers",
    "limited edition item",
    "best seller this season",
    "high quality product",
    "new arrival, great pick",
    "trusted by thousands of buyers",
]
ADJECTIVES = ["premium", "compact", "durable", "lightweight", "stylish",
              "affordable", "bestselling", "handcrafted", "modern", "classic"]
MATERIALS = {
    "Electronics": ["aluminium body", "fast charging", "wireless", "HD display"],
    "Footwear":    ["mesh upper", "rubber sole", "cushioned insole", "breathable fabric"],
    "Furniture":   ["solid wood", "metal frame", "cushioned seat", "engineered oak"],
    "Fashion":     ["cotton blend", "slim fit", "machine washable", "soft fabric"],
    "Books":       ["paperback edition", "bestseller", "illustrated", "hardcover"],
}
NOUNS = {
    "Electronics": ["wireless earbuds", "smartwatch", "bluetooth speaker", "laptop", "tablet"],
    "Footwear":    ["running shoes", "sneakers", "sandals", "loafers", "sports shoes"],
    "Furniture":   ["office chair", "dining table", "bookshelf", "sofa", "bed frame"],
    "Fashion":     ["cotton t-shirt", "denim jacket", "formal shirt", "hoodie", "kurta"],
    "Books":       ["novel", "cookbook", "biography", "textbook", "poetry collection"],
}

# fraction of samples per class where the TEXT has its category keyword
# stripped out (forcing reliance on the image) or the IMAGE is heavily
# corrupted with noise (forcing reliance on the text).
AMBIGUOUS_TEXT_FRAC = 0.25
AMBIGUOUS_IMAGE_FRAC = 0.25


def make_title(category):
    brand = random.choice(BRANDS)
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS[category])
    material = random.choice(MATERIALS[category])
    templates = [
        f"{brand} {adj} {noun} - {material}",
        f"{brand} {noun}, {adj}, {material}",
        f"{adj.capitalize()} {noun} by {brand} ({material})",
    ]
    return random.choice(templates)


def make_ambiguous_title():
    """A fully category-agnostic listing title (e.g. seller left the
    product name out of the title) -- carries NO usable category signal,
    forcing a text-only model to fail on these and rely on the image."""
    brand = random.choice(BRANDS)
    phrase = random.choice(GENERIC_PHRASES)
    return f"{brand} - {phrase}"


def jitter(v, amt):
    return max(0, min(255, int(v + random.randint(-amt, amt))))


def draw_electronics(draw, cx, cy, color, noise=0.0):
    w, h = 34, 22
    draw.rounded_rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], radius=4, fill=color)
    screen_col = tuple(jitter(c, 40) for c in (30, 30, 30))
    draw.rectangle([cx - w/2 + 4, cy - h/2 + 4, cx + w/2 - 4, cy + h/2 - 4], fill=screen_col)


def draw_footwear(draw, cx, cy, color, noise=0.0):
    pts = [(cx - 22, cy + 6), (cx - 18, cy - 8), (cx, cy - 10),
           (cx + 20, cy - 2), (cx + 22, cy + 8), (cx - 22, cy + 10)]
    draw.polygon(pts, fill=color)
    draw.line([cx - 15, cy + 8, cx + 15, cy + 9], fill=(20, 20, 20), width=2)


def draw_furniture(draw, cx, cy, color, noise=0.0):
    draw.rectangle([cx - 20, cy - 4, cx + 20, cy + 4], fill=color)
    for dx in (-16, 16):
        draw.rectangle([cx + dx - 3, cy + 4, cx + dx + 3, cy + 22], fill=color)
    draw.rectangle([cx - 20, cy - 20, cx - 14, cy + 4], fill=color)


def draw_fashion(draw, cx, cy, color, noise=0.0):
    pts = [(cx - 16, cy - 16), (cx - 6, cy - 20), (cx, cy - 14), (cx + 6, cy - 20),
           (cx + 16, cy - 16), (cx + 20, cy - 4), (cx + 12, cy - 2), (cx + 12, cy + 20),
           (cx - 12, cy + 20), (cx - 12, cy - 2), (cx - 20, cy - 4)]
    draw.polygon(pts, fill=color)


def draw_books(draw, cx, cy, color, noise=0.0):
    draw.rectangle([cx - 14, cy - 20, cx + 14, cy + 20], fill=color)
    line_col = tuple(jitter(c, 30) for c in (255, 255, 255))
    for i in range(4):
        y = cy - 12 + i * 8
        draw.line([cx - 9, y, cx + 9, y], fill=line_col, width=1)


DRAW_FN = {
    "Electronics": draw_electronics,
    "Footwear": draw_footwear,
    "Furniture": draw_furniture,
    "Fashion": draw_fashion,
    "Books": draw_books,
}

BASE_COLORS = {
    "Electronics": (60, 90, 160),
    "Footwear": (200, 90, 60),
    "Furniture": (150, 110, 70),
    "Fashion": (90, 160, 110),
    "Books": (170, 70, 130),
}


def make_ambiguous_image():
    """A fully corrupted / uninformative product photo (e.g. a bad phone
    photo, blur, wrong upload) -- pure random noise, carrying NO usable
    category signal, forcing an image-only model to fail on these and
    rely on the paired text instead."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    px = img.load()
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            px[x, y] = (random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255))
    return img


def make_image(category):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE),
                     color=tuple(jitter(c, 15) for c in (245, 245, 245)))
    draw = ImageDraw.Draw(img)
    base = BASE_COLORS[category]
    color = tuple(jitter(c, 35) for c in base)
    cx = IMG_SIZE / 2 + random.randint(-4, 4)
    cy = IMG_SIZE / 2 + random.randint(-4, 4)
    DRAW_FN[category](draw, cx, cy, color)
    return img


def main():
    rows = []
    for category in CATEGORIES:
        for i in range(N_PER_CLASS):
            amb_img = random.random() < AMBIGUOUS_IMAGE_FRAC
            amb_txt = (not amb_img) and (random.random() < AMBIGUOUS_TEXT_FRAC)
            img = make_ambiguous_image() if amb_img else make_image(category)
            fname = f"{category}_{i:04d}.png"
            fpath = os.path.join(IMG_DIR, fname)
            img.save(fpath)

            text = make_ambiguous_title() if amb_txt else make_title(category)

            rows.append({
                "image_path": os.path.join("images", fname),
                "text": text,
                "label": category,
                "ambiguous_image": int(amb_img),
                "ambiguous_text": int(amb_txt),
            })

    random.shuffle(rows)
    csv_path = os.path.join(OUT_DIR, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} paired image-text samples across {len(CATEGORIES)} categories")
    print(f"CSV: {csv_path}")
    print(f"Images: {IMG_DIR}")


if __name__ == "__main__":
    main()
