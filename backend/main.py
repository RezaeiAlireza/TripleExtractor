from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModel, BertTokenizerFast
from tplinker.tplinker import TPLinkerBert, HandshakingTaggingScheme, DataMaker4Bert
from typing import Optional
import json
from rdflib import Graph, URIRef, Literal, Namespace
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from newspaper import Article  # For extracting text from URLs
from urllib.parse import quote
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Configurations for TPLinker
config = {
    "bert_path": "./pretrained_models/bert-base-cased",
    "rel2id_path": "./data4bert/nyt_star/rel2id.json",
    "model_state_path": "./tplinker/default_log_dir/r584cHKZ/model_state_dict_2.pt",
    "max_seq_len": 100,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",  # For macOS MPS; use 'cuda' for GPU
}

# RDF namespace setup
SCHEMA = Namespace("http://schema.org/")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load relation mappings
try:
    with open(config["rel2id_path"], "r", encoding="utf-8") as f:
        rel2id = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Relation mapping file not found: {config['rel2id_path']}")

# Setup tokenizer and data maker for TPLinker
try:
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], do_lower_case=False)
    tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=config["max_seq_len"])
    data_maker = DataMaker4Bert(tokenizer, tagger)
except Exception as e:
    raise RuntimeError(f"Error loading tokenizer or data maker: {str(e)}")

# Initialize TPLinkerBert model
try:
    model = TPLinkerBert(
        encoder=AutoModel.from_pretrained(config["bert_path"]),
        rel_size=len(rel2id),
        shaking_type="cat",
        inner_enc_type="lstm",
        dist_emb_size=-1,
        ent_add_dist=False,
        rel_add_dist=False,
    ).to(config["device"])
    model.load_state_dict(torch.load(config["model_state_path"], map_location=config["device"]), strict=False)
    model.eval()  # Set model to evaluation mode
except FileNotFoundError:
    raise RuntimeError(f"Model state file not found: {config['model_state_path']}")
except Exception as e:
    raise RuntimeError(f"Error initializing model: {str(e)}")


# Data structure for incoming requests
class TextRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    format: Optional[str] = "json-ld"  # Default format


def validate_english_language(text):
    """Validate that the input text is in English."""
    try:
        language = detect(text)
        if language != 'en':
            raise ValueError(f"Input text is not in English. Detected language: {language}.")
    except LangDetectException:
        raise ValueError("Unable to detect the language of the input text.")


def fetch_text_from_url(url):
    """Fetch text content from a URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch text from URL: {str(e)}")


def extract_relations(text):
    """Extract subject-predicate-object triples from text using TPLinker."""
    try:
        if not text.strip():
            raise ValueError("Input text is empty or invalid.")
        
        sample = {"text": text, "relation_list": []}
        indexed_data = data_maker.get_indexed_data([sample], config["max_seq_len"], data_type="test")
        _, input_ids, attention_mask, token_type_ids, tok2char_span, _ = indexed_data[0]

        # Prepare tensors for model
        input_ids = input_ids.unsqueeze(0).to(config["device"])
        attention_mask = attention_mask.unsqueeze(0).to(config["device"])
        token_type_ids = token_type_ids.unsqueeze(0).to(config["device"])

        # Model inference
        with torch.no_grad():
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                input_ids, attention_mask, token_type_ids
            )

        # Decode the output to get triples
        rel_list = tagger.decode_rel_fr_shaking_tag(
            text=text,
            ent_shaking_tag=torch.argmax(ent_shaking_outputs[0], dim=-1),
            head_rel_shaking_tag=torch.argmax(head_rel_shaking_outputs[0], dim=-1),
            tail_rel_shaking_tag=torch.argmax(tail_rel_shaking_outputs[0], dim=-1),
            tok2char_span=tok2char_span,
        )
        return rel_list
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise RuntimeError(f"Error during relation extraction: {str(e)}")


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
        output += f"{triple['subject']},{triple['predicate']},{triple['object']}\n"
    return output


def serialize_to_rdf(triples):
    """Serialize triples to RDF format with URI sanitization."""
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


@app.post("/extract-triples/")
async def extract_triples(request: TextRequest):
    """API endpoint to extract triples and return in the requested format."""
    try:
        if request.url and request.text:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'url', not both.")
        if not request.url and not request.text:
            raise HTTPException(status_code=400, detail="Either 'text' or 'url' must be provided.")

        # Fetch text from URL if needed
        if request.url:
            text = fetch_text_from_url(request.url)
        else:
            text = request.text.strip()

        # Validate that the text is in English
        validate_english_language(text)

        # Extract relations
        triples = extract_relations(text)

        # Serialize results
        format = request.format.lower()
        if format == "json-ld":
            content = json.dumps(serialize_to_jsonld(triples), indent=2)
            filename = "output.jsonld"
            content_type = "application/ld+json"
        elif format == "csv":
            content = serialize_to_csv(triples)
            filename = "output.csv"
            content_type = "text/csv"
        elif format == "rdf":
            content = serialize_to_rdf(triples)
            filename = "output.ttl"
            content_type = "text/turtle"
        elif format == "xml":
            content = serialize_to_xml(triples)
            filename = "output.xml"
            content_type = "application/xml"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format.")

        # Return downloadable response
        response = PlainTextResponse(content=content, media_type=content_type)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "TPLinker Relation Extraction API"}
