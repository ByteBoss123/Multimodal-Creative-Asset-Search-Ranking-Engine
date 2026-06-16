"""
corpus.py
---------
Builds a searchable asset corpus from the real MS COCO val2014 dataset.

Each COCO image becomes one "asset" with:
  - asset_id     : COCO image ID (string)
  - image_url    : Flickr URL of the original image
  - captions     : list of up to 5 human-written captions
  - description  : all captions joined — used for CLIP embedding
  - tags         : noun/keyword tokens extracted from captions
  - category     : coarse visual category inferred from caption vocabulary
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

CATEGORY_KEYWORDS = {
    "Food & Drink":   ["food","eating","pizza","sandwich","cake","fruit","vegetable",
                       "kitchen","cooking","restaurant","meal","drink","coffee","wine",
                       "banana","apple","hot dog","donut","broccoli","carrot","bowl"],
    "Animals":        ["dog","cat","bird","horse","cow","sheep","elephant","bear",
                       "zebra","giraffe","animal","wildlife","pet","puppy","kitten"],
    "Sports":         ["sport","playing","game","ball","field","court","team",
                       "baseball","football","tennis","soccer","basketball","skiing",
                       "surfing","skateboard","frisbee","kite","snowboard"],
    "Transportation": ["car","truck","bus","train","plane","airplane","boat",
                       "motorcycle","bicycle","bike","vehicle","traffic","street",
                       "road","driving","airport","station"],
    "People":         ["person","people","man","woman","child","boy","girl",
                       "crowd","group","standing","sitting","walking","holding"],
    "Nature":         ["outdoor","sky","tree","mountain","beach","water","ocean",
                       "lake","river","forest","field","grass","flower","park",
                       "nature","sunset","snow","cloud"],
    "Urban":          ["city","building","street","sidewalk","downtown","urban",
                       "store","shop","sign","window","bridge","tower"],
    "Home & Indoor":  ["room","living","bedroom","bathroom","table","chair","desk",
                       "couch","sofa","floor","wall","furniture","indoor","inside","home"],
    "Technology":     ["computer","phone","laptop","screen","keyboard","device",
                       "television","monitor","camera","electronic"],
}

EVAL_QUERIES = [
    "a dog playing in the park",
    "people eating food at a restaurant",
    "a city street with cars and traffic",
    "a person riding a bicycle",
    "children playing sports outdoors",
    "a woman sitting at a table",
    "a group of people at a beach",
    "an airplane flying in the sky",
    "a cat sitting on a couch",
    "a kitchen with food on the counter",
]

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","in","on","at",
    "to","for","of","with","by","from","as","and","or","but","that","this",
    "it","its","there","their","some","has","have","had","not","no","into",
    "onto","over","two","three","four","five","one","several","many","next",
    "while","near","front","back","top","side","each","very","which","who",
    "what","where","when","how","can","could","would","should","will","may",
    "might","do","does","did","another","other","than","around","behind",
    "between","through","also","just","then","them","they","she","he","his",
    "her","its","our","your","my","we","us","you","after","before","during",
}


def _infer_category(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower)
              for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def _extract_tags(captions: list) -> list:
    counts = defaultdict(int)
    for cap in captions:
        for w in re.findall(r"[a-z]+", cap.lower()):
            if len(w) > 3 and w not in STOPWORDS:
                counts[w] += 1
    return [w for w, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 2][:15]


def _relevance(query: str, description: str, tags: list) -> float:
    q = set(re.findall(r"[a-z]+", query.lower()))
    a = set(re.findall(r"[a-z]+", description.lower())) | set(tags)
    return min(1.0, len(q & a) / max(len(q), 1))


def build_corpus(coco_path: str, output_path: str, limit: int = None) -> list:
    print(f"Loading {coco_path}...")
    with open(coco_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    caps_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])

    ids = list(images.keys())
    if limit:
        ids = ids[:limit]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    assets = []
    print(f"Building corpus: {len(ids):,} images...")

    with open(output_path, "w") as f:
        for img_id in tqdm(ids, desc="Building"):
            img = images[img_id]
            captions = caps_by_img.get(img_id, [])
            if not captions:
                continue

            description = " | ".join(captions)
            tags = _extract_tags(captions)
            category = _infer_category(description)

            asset = {
                "asset_id":    str(img_id),
                "image_url":   img.get("url", ""),
                "file_name":   img.get("file_name", ""),
                "width":       img.get("width", 0),
                "height":      img.get("height", 0),
                "captions":    captions,
                "description": description,
                "tags":        tags,
                "category":    category,
                "relevance_labels": {
                    q: _relevance(q, description, tags) for q in EVAL_QUERIES
                },
            }
            assets.append(asset)
            f.write(json.dumps(asset) + "\n")

    print(f"Done: {len(assets):,} assets → {output_path}")
    return assets


def load_asset_lookup(path: str) -> dict:
    lookup = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                a = json.loads(line)
                lookup[a["asset_id"]] = a
    return lookup
