#!/usr/bin/env python3
import argparse, json, os, sys
from typing import List, Dict, Any, Optional

def parse_feats_str(feats_str: str) -> Dict[str, str]:
    if not feats_str or feats_str == "_" or feats_str.strip() == "":
        return {}
    out = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

def read_conllu(path: str) -> List[Dict[str, Any]]:
    """
    Parse a UD CoNLL-U file into a list of sentence dicts:
      {tokens, upos, heads, rels, feats, (optional id from sent_id)}
    Skips multiword tokens (ID like '1-2') and empty nodes (ID like '1.1').
    """
    sents = []
    tokens: List[str] = []
    upos: List[str] = []
    heads: List[int] = []
    rels: List[str] = []
    feats_list: List[str] = []
    cur_sent_id: Optional[str] = None

    def flush():
        nonlocal tokens, upos, heads, rels, feats_list, cur_sent_id
        if tokens:
            ex = {
                "tokens": tokens,
                "upos": upos,
                "heads": heads,
                "rels": rels,
                "feats": feats_list,
            }
            if cur_sent_id is not None:
                ex["id"] = cur_sent_id
            sents.append(ex)
        tokens, upos, heads, rels, feats_list = [], [], [], [], []
        cur_sent_id = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                # capture sent_id if present
                if line.startswith("# sent_id"):
                    # formats: "# sent_id = xyz" or "# sent_id: xyz"
                    if "=" in line:
                        cur_sent_id = line.split("=", 1)[1].strip()
                    elif ":" in line:
                        cur_sent_id = line.split(":", 1)[1].strip()
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                # Malformed line; skip or raise
                continue
            tid = cols[0]
            # Skip multiword tokens (e.g., "1-2") and empty nodes (e.g., "3.1")
            if "-" in tid or "." in tid:
                continue
            form = cols[1]
            up = cols[3]
            feats = cols[5]
            try:
                head = int(cols[6])
            except ValueError:
                head = 0
            rel = cols[7]

            tokens.append(form)
            upos.append(up)
            heads.append(head)
            rels.append(rel)
            feats_list.append(feats)

    # flush last
    flush()
    return sents

def reconstruct_text(tokens: List[str]) -> str:
    # Simple whitespace join; replace with language-specific detok if needed.
    return " ".join(tokens)

def convert_to_jsonl_objs(
    sents: List[Dict[str, Any]],
    treebank: Optional[str],
    split: Optional[str],
    add_text: bool,
) -> List[Dict[str, Any]]:
    out = []
    for idx, ex in enumerate(sents):
        n = len(ex["tokens"])
        # basic validation
        if not (len(ex["upos"]) == len(ex["heads"]) == len(ex["rels"]) == len(ex["feats"]) == n):
            raise ValueError(f"Sentence {idx} has inconsistent lengths: "
                             f"tokens={n} upos={len(ex['upos'])} heads={len(ex['heads'])} "
                             f"rels={len(ex['rels'])} feats={len(ex['feats'])}")

        feats_dict = [parse_feats_str(s) for s in ex["feats"]]
        obj = {
            "id": ex.get("id", idx),
            "tokens": ex["tokens"],
            "upos": ex["upos"],
            "heads": ex["heads"],
            "rels": ex["rels"],
            "feats": ex["feats"],
            "feats_dict": feats_dict,
        }
        if add_text:
            obj["text"] = reconstruct_text(ex["tokens"])
        if treebank is not None:
            obj["treebank"] = treebank
        if split is not None:
            obj["split"] = split
        out.append(obj)
    return out

def write_jsonl(items: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def convert_file(in_path: str, out_path: str, treebank: str, split: str, add_text: bool):
    sents = read_conllu(in_path)
    objs = convert_to_jsonl_objs(sents, treebank=treebank, split=split, add_text=add_text)
    write_jsonl(objs, out_path)
    print(f"Wrote {len(objs)} examples to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert UD CoNLL-U to Hugging Face JSONL (one JSON per line).")
    ap.add_argument("--train", type=str, help="Path to train.conllu")
    ap.add_argument("--dev", type=str, help="Path to dev.conllu")
    ap.add_argument("--test", type=str, help="Path to test.conllu")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for JSONL files")
    ap.add_argument("--treebank", type=str, required=True, help="Treebank tag to include (e.g., sabanci, boun_imst)")
    ap.add_argument("--prefix", type=str, default="", help="Optional file prefix, e.g., 'sabanci_'")
    ap.add_argument("--no_text", action="store_true", help="Do not include reconstructed 'text' field")
    args = ap.parse_args()

    add_text = not args.no_text
    prefix = (args.prefix + "_") if args.prefix else ""

    if not any([args.train, args.dev, args.test]):
        print("Provide at least one of --train/--dev/--test", file=sys.stderr)
        sys.exit(1)

    if args.train:
        convert_file(args.train, os.path.join(args.out_dir, f"{prefix}train.jsonl"),
                     treebank=args.treebank, split="train", add_text=add_text)
    if args.dev:
        convert_file(args.dev, os.path.join(args.out_dir, f"{prefix}dev.jsonl"),
                     treebank=args.treebank, split="dev", add_text=add_text)
    if args.test:
        convert_file(args.test, os.path.join(args.out_dir, f"{prefix}test.jsonl"),
                     treebank=args.treebank, split="test", add_text=add_text)

if __name__ == "__main__":
    main()
