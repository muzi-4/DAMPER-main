import re
from typing import List, Optional

try:
    import spacy
    _SPACY_OK = True
except Exception:
    spacy = None
    _SPACY_OK = False


class TextChunker:

    def __init__(self, spacy_model: str = "en_core_web_sm",
                 nlp: Optional["spacy.language.Language"] = None):
        if nlp is not None:
            self.nlp = nlp
        else:
            if not _SPACY_OK:
                raise RuntimeError(
                    "spaCy is not installed. Please first `pip install spacy`, and download the model: "
                    "`python -m spacy download en_core_web_sm`"
                )
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception as e:
                raise RuntimeError(
                    f"Unable to load spaCy model '{spacy_model}'. Please run: "
                    f"python -m spacy download {spacy_model}"
                ) from e

        # === Precompile regex ===
        frame_patts = [
            r"\b(history of)\b",                    # medical
            r"\b(symptoms of)\b",
            r"\b(presents|reports|complains of)\b",
            r"\b(protected by)\b",                  # legal
            r"\b(infringed upon)\b",
            r"\b(contains|consisted of|consists of)\b",
            r"\b(is to)\b",                         # purpose: is to V...
        ]
        self.FRAME_RE = re.compile("|".join(frame_patts), flags=re.I)
        self.WEAK_SEP_RE = re.compile(r"\s*(,|;|，|；)\s*", flags=re.I)
        self.CONJ_RE = re.compile(r"^\s*(and|or)\s*$", flags=re.I)

        # Internal sentence punctuation: not allowed inside final spans
        self.INNER_PUNCT_RE = re.compile(r"[，,;；。\.!?？]")

    # ---------- Basic cleaning: only outer trimming ----------
    @staticmethod
    def _clean_span(text: str) -> str:

        s = text.strip()
        s = re.sub(r"[\s,;，；\.]+$", "", s)
        return s

    # ---------- Remove leading conjunctions ----------
    @staticmethod
    def _strip_leading_conj(text: str) -> str:

        return re.sub(r"^(and|or)\s+", "", text, flags=re.I)

    # ---------- Enumeration splitting after frame ----------
    def _enumeration_chunks(self, after_text: str) -> List[str]:

        parts = self.WEAK_SEP_RE.split(after_text)
        segs, buf = [], []

        for piece in parts:
            if piece in {",", ";", "，", "；"}:
                # True separator: end a segment
                if buf:
                    segs.append("".join(buf))
                    buf = []
            else:
                # Conjunctions and/or/以及/并且: also act as soft boundaries
                if self.CONJ_RE.match(piece):
                    if buf:
                        segs.append("".join(buf))
                        buf = []
                else:
                    buf.append(piece)
        if buf:
            segs.append("".join(buf))

        out = []
        for s in segs:
            s = self._clean_span(s)
            if s:
                out.append(s)
        return out

    # ---------- Intra-sentence NP/VP supplement (depends on spaCy) ----------
    def _extract_vp_np_phrases(self, sent_doc) -> List[str]:

        spans = set()
        doc = sent_doc.doc  # sent_doc is a Span, doc is the whole document

        # 1) spaCy noun chunks
        for nc in sent_doc.noun_chunks:
            s = self._clean_span(nc.text)
            if s:
                spans.add(s)

        # 2) VBG subtrees (e.g., carrying/selling ...)
        for tok in sent_doc:
            if tok.tag_ == "VBG":
                subtree_tokens = list(tok.subtree)
                if not subtree_tokens:
                    continue
                start = min(t.i for t in subtree_tokens)
                end = max(t.i for t in subtree_tokens) + 1
                raw = doc[start:end].text
                s = self._clean_span(raw)
                # If too long, truncate to next separator
                if len(s.split()) > 8:
                    right_text = doc[tok.i: end].text
                    m = re.search(r"(.+?)(,|;|，|；|\.|$)", right_text)
                    if m:
                        s = self._clean_span(m.group(1))
                if s:
                    spans.add(s)

        # 3) Infinitive purpose to V ...
        for tok in sent_doc:
            if tok.text.lower() == "to":
                right_text = doc[tok.i: sent_doc[-1].i + 1].text
                m = re.search(
                    r"(to\s+[A-Za-z]+(?:\s+\w+){0,6}?)(?:,|;|，|；|\.|$)",
                    right_text,
                    flags=re.I,
                )
                if m:
                    s = self._clean_span(m.group(1))
                    if s:
                        spans.add(s)

        return list(spans)

    # ---------- Split by internal punctuation again ----------
    def _split_inner_punct(self, spans: List[str]) -> List[str]:

        results: List[str] = []
        for s in spans:
            # First split by internal sentence punctuation
            parts = self.INNER_PUNCT_RE.split(s)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Do another simple trailing punctuation cleanup to ensure clean
                p = self._clean_span(p)
                if p:
                    results.append(p)
        return results

    # ---------- Restore surface forms based on original text ----------
    @staticmethod
    def _restore_surface_forms(text: str, spans: List[str]) -> List[str]:

        restored = []
        for s in spans:
            if not s:
                continue
            m = re.search(re.escape(s), text)
            if m:
                # ✅ Need parentheses here: start() / end() return int
                restored.append(text[m.start():m.end()])
            else:
                # Should not happen theoretically, fallback
                restored.append(s)
        return restored

    # ========== Main public interface ==========
    def chunk_text(self, text: str) -> List[str]:

        doc = self.nlp(text)
        raw_spans: List[str] = []

        for sent in doc.sents:
            s_text = sent.text.strip()

            # Frame trigger → first enumerate and split
            m = self.FRAME_RE.search(s_text)
            if m:
                after_text = s_text[m.end():]
                items = self._enumeration_chunks(after_text)
                raw_spans.extend(items)

            # Supplement: NP/VP (VBG, infinitive)
            extra = self._extract_vp_np_phrases(sent)
            raw_spans.extend(extra)

        # First do basic clean + remove leading conjunctions (still maintaining as sub-intervals of original substring)
        cleaned_spans: List[str] = []
        for s in raw_spans:
            s1 = self._clean_span(s)
            s2 = self._strip_leading_conj(s1)
            s2 = s2.strip()
            if s2:
                cleaned_spans.append(s2)

        # New rule: split again by internal punctuation, prohibit internal sentence punctuation
        splitted_spans = self._split_inner_punct(cleaned_spans)

        # Deduplicate (case-insensitive)
        seen, deduped = set(), []
        for s in splitted_spans:
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                deduped.append(s)

        # Key: restore once in original text to ensure it's a real substring of text
        deduped = self._restore_surface_forms(text, deduped)
        return deduped


# Convenience function: maintain your original functional interface
def chunk_text(text: str) -> List[str]:

    chunker = TextChunker()
    return chunker.chunk_text(text)
