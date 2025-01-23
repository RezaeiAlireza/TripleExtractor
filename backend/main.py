import torch
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from newspaper import Article
from langdetect import detect, LangDetectException
from urllib.parse import quote
from rdflib import Graph, URIRef, Literal, Namespace
from transformers import AutoModel, BertTokenizerFast
from typing import Optional
import spacy
nlp = spacy.load("en_core_web_sm")

from tplinker.tplinker import (
    TPLinkerBert,
    HandshakingTaggingScheme,
    DataMaker4Bert
)

# Configuration
config = {
    # NYT (BERT)
    "bert_path_nyt": "./pretrained_models/bert-base-cased",
    "rel2id_path_nyt": "./data4bert/nyt/rel2id.json",
    "model_state_path_nyt": "./tplinker/default_log_dir/nyt/model_state_dict_0.pt",

    # WebNLG (BERT)
    "bert_path_webnlg": "./pretrained_models/bert-base-cased",
    "rel2id_path_webnlg": "./data4bert/webnlg/rel2id.json",
    "model_state_path_webnlg": "./tplinker/default_log_dir/webnlg/model_state_dict_33.pt",

    "max_seq_len": 100,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}

SCHEMA = Namespace("http://schema.org/")

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------
# NYT (BERT) Model Initialization
# --------------
try:
    with open(config["rel2id_path_nyt"], "r", encoding="utf-8") as f:
        rel2id_nyt = json.load(f)
    tokenizer_nyt = BertTokenizerFast.from_pretrained(config["bert_path_nyt"], do_lower_case=False)
    nyt_tagger = HandshakingTaggingScheme(rel2id=rel2id_nyt, max_seq_len=config["max_seq_len"])
    nyt_data_maker = DataMaker4Bert(tokenizer_nyt, nyt_tagger)

    nyt_model = TPLinkerBert(
        encoder=AutoModel.from_pretrained(config["bert_path_nyt"]),
        rel_size=len(rel2id_nyt),
        shaking_type="cln",
        inner_enc_type="lstm",
        dist_emb_size=-1,
        ent_add_dist=False,
        rel_add_dist=False,
    ).to(config["device"])
    nyt_model.load_state_dict(
        torch.load(config["model_state_path_nyt"], map_location=config["device"], weights_only=True),
        strict=False
    )
    nyt_model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading NYT (BERT) model: {str(e)}")

# --------------
# WebNLG (BERT) Model Initialization
# --------------
try:
    with open(config["rel2id_path_webnlg"], "r", encoding="utf-8") as f:
        rel2id_webnlg = json.load(f)
    tokenizer_webnlg = BertTokenizerFast.from_pretrained(config["bert_path_webnlg"], do_lower_case=False)
    webnlg_tagger = HandshakingTaggingScheme(rel2id=rel2id_webnlg, max_seq_len=config["max_seq_len"])
    webnlg_data_maker = DataMaker4Bert(tokenizer_webnlg, webnlg_tagger)

    webnlg_model = TPLinkerBert(
        encoder=AutoModel.from_pretrained(config["bert_path_webnlg"]),
        rel_size=len(rel2id_webnlg),
        shaking_type="cln",
        inner_enc_type="lstm",
        dist_emb_size=-1,
        ent_add_dist=False,
        rel_add_dist=False,
    ).to(config["device"])
    webnlg_model.load_state_dict(
        torch.load(config["model_state_path_webnlg"], map_location=config["device"], weights_only=True),
        strict=False
    )
    webnlg_model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading WebNLG (BERT) model: {str(e)}")

# -----------------------------
# Helper Classes & Functions
# -----------------------------
class TextRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    format: Optional[str] = "json-ld"
    model: Optional[str] = "NYT"


def validate_english_language(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def fetch_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text.strip()


def filter_duplicate_sentences(text: str) -> list:
    raw_sentences = text.split('\n')
    seen = set()
    unique_sentences = []
    for line in raw_sentences:
        if line.strip():
            doc = nlp(line.strip())
            for sent in doc.sents:
                sentence_str = sent.text.strip()
                if sentence_str not in seen and sentence_str != "":
                    unique_sentences.append(sentence_str)
                    seen.add(sentence_str)
    return unique_sentences



def extract_relations(text, model_choice):
    """
    Extract subject-predicate-object triples using the specified model (NYT or WebNLG).
    """
    sample = {"text": text, "relation_list": []}

    if model_choice == "WEBNLG":
        indexed_data = webnlg_data_maker.get_indexed_data([sample], config["max_seq_len"], data_type="test")
        _, input_ids, attention_mask, token_type_ids, tok2char_span, _ = indexed_data[0]

        input_ids = input_ids.unsqueeze(0).to(config["device"])
        attention_mask = attention_mask.unsqueeze(0).to(config["device"])
        token_type_ids = token_type_ids.unsqueeze(0).to(config["device"])

        with torch.no_grad():
            ent_outputs, head_rel_outputs, tail_rel_outputs = webnlg_model(input_ids, attention_mask, token_type_ids)

        rel_list = webnlg_tagger.decode_rel_fr_shaking_tag(
            text=text,
            ent_shaking_tag=torch.argmax(ent_outputs[0], dim=-1),
            head_rel_shaking_tag=torch.argmax(head_rel_outputs[0], dim=-1),
            tail_rel_shaking_tag=torch.argmax(tail_rel_outputs[0], dim=-1),
            tok2char_span=tok2char_span,
        )
    else:
        indexed_data = nyt_data_maker.get_indexed_data([sample], config["max_seq_len"], data_type="test")
        _, input_ids, attention_mask, token_type_ids, tok2char_span, _ = indexed_data[0]

        input_ids = input_ids.unsqueeze(0).to(config["device"])
        attention_mask = attention_mask.unsqueeze(0).to(config["device"])
        token_type_ids = token_type_ids.unsqueeze(0).to(config["device"])

        with torch.no_grad():
            ent_outputs, head_rel_outputs, tail_rel_outputs = nyt_model(input_ids, attention_mask, token_type_ids)

        rel_list = nyt_tagger.decode_rel_fr_shaking_tag(
            text=text,
            ent_shaking_tag=torch.argmax(ent_outputs[0], dim=-1),
            head_rel_shaking_tag=torch.argmax(head_rel_outputs[0], dim=-1),
            tail_rel_shaking_tag=torch.argmax(tail_rel_outputs[0], dim=-1),
            tok2char_span=tok2char_span,
        )

    return rel_list


def serialize_to_jsonld(triples):
    """Serialize triples to JSON-LD format."""
    return {
        "@context": {
            "subject": "http://schema.org/subject",
            "predicate": "http://schema.org/predicate",
            "object": "http://schema.org/object",
        },
        "@graph": [
            {
                "@type": "Triple",
                "subject": triple["subject"],
                "predicate": triple["predicate"],
                "object": triple["object"],
            }
            for triple in triples
        ],
    }


def serialize_to_csv(triples):
    """Serialize triples to CSV format."""
    output = "subject,predicate,object\n"
    for triple in triples:
        subject = triple["subject"].replace(",", " ")
        predicate = triple["predicate"].replace(",", " ")
        obj = triple["object"].replace(",", " ")
        output += f"{subject},{predicate},{obj}\n"
    return output


def serialize_to_rdf(triples):
    """Serialize triples to RDF format."""
    g = Graph()
    for triple in triples:
        subject = URIRef(SCHEMA[quote(triple["subject"].replace(" ", "_"))])
        predicate = URIRef(SCHEMA[quote(triple["predicate"].replace(" ", "_"))])
        obj = Literal(triple["object"].strip())
        g.add((subject, predicate, obj))
    return g.serialize(format="turtle")


def serialize_to_xml(triples):
    """Serialize triples to XML format."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    root = Element("Triples")
    for triple in triples:
        triple_el = SubElement(root, "Triple")
        SubElement(triple_el, "Subject").text = triple["subject"]
        SubElement(triple_el, "Predicate").text = triple["predicate"]
        SubElement(triple_el, "Object").text = triple["object"]
    return tostring(root, encoding="unicode")


# --------------
# Endpoint to Extract Triples
# --------------
@app.post("/extract-triples/")
async def extract_triples(request: TextRequest):
    """
    API endpoint to extract triples and return in the requested format.
    This endpoint filters out duplicate sentences before passing them to the model.
    """
    model_choice = request.model.upper()
    not_supported_inputs = []
    try:
        # Validate inputs
        if request.url and request.text:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'text' or 'url', not both."
            )
        if not request.url and not request.text:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'url' must be provided."
            )

        if request.url:
            text = fetch_text_from_url(request.url)
        else:
            text = request.text.strip()
        rel_list_temp=[]
        text = filter_duplicate_sentences(text)
        for i in text:
            if validate_english_language(i):
                rel_list_temp.append(extract_relations(i, model_choice))
            else:
                not_supported_inputs.append(i)
        rel_list=[]
        for i in rel_list_temp:
            for j in i:
                rel_list.append(j)
        if not_supported_inputs:
            with open("not_extracted_inputs.txt", "w", encoding="utf-8") as file:
                file.write("\n".join(not_supported_inputs))
        fmt = request.format.lower()
        if fmt == "json-ld":
            serialized_output = serialize_to_jsonld(rel_list)
            content = json.dumps(serialized_output, indent=2)
            filename = "output.jsonld"
            content_type = "application/ld+json"
        elif fmt == "csv":
            content = serialize_to_csv(rel_list)
            filename = "output.csv"
            content_type = "text/csv"
        elif fmt == "rdf":
            content = serialize_to_rdf(rel_list)
            filename = "output.ttl"
            content_type = "text/turtle"
        elif fmt == "xml":
            content = serialize_to_xml(rel_list)
            filename = "output.xml"
            content_type = "application/xml"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format.")

        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/download-not-extracted/")
async def download_not_extracted():
    """Endpoint to download non-English input texts."""
    file_path = "not_extracted_inputs.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=not_extracted_inputs.txt"}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No non-English inputs found.")


# --------------
# Health Check Endpoint
# --------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Relation Extraction API is running"}
