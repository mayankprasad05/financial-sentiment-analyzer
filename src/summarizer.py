"""
Section-aware, Loughran-McDonald-guided abstractive summarization.

This module holds the project's single summarizer, ``TranscriptSummarizer``. It replaces
the previous "truncate to 1024 tokens and call BART" behavior with a five-stage
pipeline:

1. **Strip boilerplate** - drop safe-harbor / forward-looking-statement sentences and
   opening call logistics, which otherwise dominate the first ~750 words of every
   earnings call (and therefore dominated every previous summary, because the model
   only ever saw that prefix).
2. **Split sections** - separate prepared remarks from the Q&A session.
3. **Chunk** - split each section into token-budgeted chunks on sentence boundaries.
4. **Rank and select** - score sentences with the Loughran-McDonald dictionary and a
   numeric-density signal, rank chunks, and pick the highest-signal chunks within the
   model input budget while guaranteeing at least one chunk per section.
5. **Map-reduce generate** - summarize each selected chunk, then summarize the chunk
   summaries into the final executive summary.

Both heuristics are configurable, which is what lets the evaluation harness run
ablations (and reconstruct the legacy truncation baseline) without duplicating code.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.loughran_mcdonald import LoughranMcDonaldScorer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/bart-large-cnn"

# --- Pattern banks ---------------------------------------------------------------
# A sentence matching any STRONG pattern is boilerplate on its own. WEAK patterns need
# two independent hits, so a single incidental "SEC" or "press release" cannot delete
# a substantive sentence.
_STRONG_BOILERPLATE = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bforward[- ]looking statements?\b",
    r"\bsafe harbou?r\b",
    r"\bactual results (?:may|could|might|will) differ\b",
    r"\bdiffer materially from\b",
    r"\bund(?:er|ertake|ertakes) no obligation\b",
    r"\bno obligation to (?:publicly )?(?:update|revise)\b",
    r"\brisks and uncertainties\b",
    r"\bshould not be (?:relied|unduly relied) upon\b",
    r"\bexcept as required by (?:applicable )?law\b",
    r"\b(?:do|does) not undertake any duty\b",
    r"\bqualified in (?:their|its) entirety by\b",
    r"\bincluding,? without limitation,? (?:those )?(?:regarding|relating to)\b",
)]

_WEAK_BOILERPLATE = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bsec\b",
    r"\bsecurities and exchange commission\b",
    r"\bform 10-[kq]\b",
    r"\bannual report\b",
    r"\bquarterly report\b",
    r"\bpress release\b",
    r"\bearnings release\b",
    r"\binvestor relations\b",
    r"\bconference call\b",
    r"\bregulation g\b",
    r"\bdisclaimer\b",
    r"\bunforeseen\b",
    r"\bfrom time to time\b",
)]

# Call logistics / housekeeping that appear before the actual content starts.
_PREAMBLE_DISCOURSE = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"^welcome to\b",
    r"^good (?:morning|afternoon|evening)[,.]?\b",
    r"^thank you(?:,| and welcome)?\b",
    r"^thanks(?:,| and welcome)?\b",
    r"^my name is\b",
    r"\b(?:this|today's) call is being recorded\b",
    r"^please note that\b",
    r"^as a reminder\b",
    r"\bdirector of investor relations\b",
    r"\bhead of investor relations\b",
    r"\boperator\b.*\b(?:instructions|conference)\b",
    r"\bwill (?:now )?(?:begin|open) the (?:question|call)\b",
    r"^after that\b",
    r"\bspeaking (?:first|next) today\b",
    r"\b(?:we'll|we will|let's) (?:now )?open (?:up )?the (?:call|lines?) (?:to|for) questions?\b",
    r"^our next question is from\b",
    r"^operator,\b",
    r"^thanks?(?: you)? for taking (?:my|our) questions?\b",
)]

_QA_START = re.compile(
    r"\b(?:question[- ]and[- ]answer|q\s*&\s*a|open (?:up )?the (?:call|lines?) (?:to|for) questions?"
    r"|begin the question|take (?:our|the) first question|first question (?:comes|is) from"
    r"|next question is from|we(?:'ll| will) (?:now )?(?:go ahead and )?take (?:our|the) (?:first|next) question)\b",
    re.IGNORECASE,
)

# The opening agenda usually announces "we'll open the call to questions" long before the
# Q&A actually starts, so candidates inside this leading window are ignored.
_QA_MIN_POSITION = 0.10

_SECTION_LABELS = {
    "prepared_remarks": "Prepared remarks",
    "qa": "Q&A",
    "transcript": "Summary",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_NUMERIC_TOKEN = re.compile(
    r"(\$?\d[\d,\.]*\s?(?:%|percent|bps|basis points|x|million|billion|trillion)?"
    r"|\b(?:million|billion|trillion|basis points|bps)\b)",
    re.IGNORECASE,
)


@dataclass
class SummarizerConfig:
    """Knobs for the summarization pipeline.

    Setting ``strip_boilerplate=False``, ``select_by_salience=False``,
    ``chunk_tokens=1024`` and ``max_input_tokens=1024`` reproduces the legacy
    "truncate and summarize" baseline that the harness compares against.
    """

    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None

    # Input budget
    chunk_tokens: int = 900
    max_input_tokens: int = 4800
    max_boilerplate_fraction: float = 0.35
    # Share of the budget reserved for prepared remarks when a Q&A section exists:
    # the prepared remarks are the company's own account of results and guidance, so
    # they get covered before the (usually much longer) Q&A session.
    prepared_remarks_share: float = 0.6

    # Selection
    strip_boilerplate: bool = True
    select_by_salience: bool = True
    numeric_weight: float = 0.6
    sentiment_weight: float = 1.2
    hedging_weight: float = 0.9

    # Generation
    chunk_summary_tokens: int = 110
    chunk_summary_min_tokens: int = 30
    max_summary_tokens: int = 190
    min_summary_tokens: int = 85
    num_beams: int = 4
    # Sentences kept from each chunk summary before the synthesis pass. Keeping a few
    # per chunk (rather than everything) is what stops the synthesis from collapsing
    # onto whichever topic happened to come first.
    synthesis_sentences_per_chunk: int = 2


@dataclass
class SentenceRecord:
    text: str
    section: str
    salience: float
    tokens: int
    numeric_hits: int
    lm_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    index: int
    section: str
    sentences: List[SentenceRecord] = field(default_factory=list)
    tokens: int = 0
    salience: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)


class TranscriptSummarizer:
    """Section-aware, LM-guided abstractive summarizer for earnings transcripts."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        lm_scorer: Optional[LoughranMcDonaldScorer] = None,
        config: Optional[SummarizerConfig] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        if config is None:
            config = SummarizerConfig(model_name=model_name, device=device)
        else:
            config.model_name = model_name or config.model_name
            if device is not None:
                config.device = device

        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Tokenizer and model are lazy; passing them in skips the download entirely
        # (used for local fine-tuned artifacts and for structural tests).
        self._tokenizer = tokenizer
        self._model = model
        self._lm_scorer = lm_scorer
        logger.info(
            "Initializing TranscriptSummarizer (model=%s, device=%s, strip_boilerplate=%s, "
            "salience_selection=%s, chunk_tokens=%d, budget=%d)",
            self.config.model_name,
            self.device,
            self.config.strip_boilerplate,
            self.config.select_by_salience,
            self.config.chunk_tokens,
            self.config.max_input_tokens,
        )

    # -- lazy resources ---------------------------------------------------------
    @property
    def lm_scorer(self) -> LoughranMcDonaldScorer:
        if self._lm_scorer is None:
            self._lm_scorer = LoughranMcDonaldScorer()
        return self._lm_scorer

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            logger.info("Loading summarizer tokenizer: %s", self.config.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    def _ensure_model(self):
        if self._model is None:
            self._ensure_tokenizer()
            logger.info("Loading summarization model: %s", self.config.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self._model.eval()
            logger.info("Summarization model loaded on %s", self.device)
        return self._model

    def _input_limit(self) -> int:
        tokenizer = self._ensure_tokenizer()
        limit = getattr(tokenizer, "model_max_length", 1024)
        if not isinstance(limit, int) or limit <= 0 or limit > 100_000:
            limit = 1024
        return limit

    def _token_len(self, text: str) -> int:
        tokenizer = self._ensure_tokenizer()
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def token_count(self, text: str) -> int:
        """Public token count using this summarizer's tokenizer."""
        return self._token_len(text)

    # -- stage 1: boilerplate ---------------------------------------------------
    def strip_boilerplate(self, sentences: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Remove safe-harbor and opening-logistics sentences.

        Returns ``(kept, removed)``. Removal stops at ``max_boilerplate_fraction`` so a
        pattern that over-triggers can never delete most of a transcript.
        """
        if not sentences or not self.config.strip_boilerplate:
            return list(sentences), []

        cap = max(1, int(len(sentences) * self.config.max_boilerplate_fraction))
        preamble_window = max(3, int(len(sentences) * 0.12))

        kept: List[str] = []
        removed: List[str] = []
        for index, sentence in enumerate(sentences):
            if len(removed) >= cap:
                kept.append(sentence)
                continue

            if any(pattern.search(sentence) for pattern in _STRONG_BOILERPLATE):
                removed.append(sentence)
                continue

            weak_hits = sum(
                1 for pattern in _WEAK_BOILERPLATE if pattern.search(sentence)
            )
            if weak_hits >= 2:
                removed.append(sentence)
                continue

            if index < preamble_window and any(
                pattern.search(sentence) for pattern in _PREAMBLE_DISCOURSE
            ):
                removed.append(sentence)
                continue

            kept.append(sentence)

        if removed:
            logger.info(
                "Stripped %d boilerplate sentences (cap=%d, %d kept)",
                len(removed),
                cap,
                len(kept),
            )
        return kept, removed

    # -- stage 2: sections ------------------------------------------------------
    def split_sections(self, sentences: Sequence[str]) -> List[Tuple[str, List[str]]]:
        """Split sentences into prepared remarks and the Q&A session.

        The earliest Q&A marker that is past the opening agenda window wins, and it must
        leave a substantial tail behind it; otherwise the transcript is left unsplit.
        """
        if not sentences:
            return []

        total = len(sentences)
        min_index = max(6, int(total * _QA_MIN_POSITION))
        min_tail = max(3, int(total * 0.15))

        for index, sentence in enumerate(sentences):
            if index < min_index or total - index < min_tail:
                continue
            if _QA_START.search(sentence):
                prepared = list(sentences[:index])
                qa = list(sentences[index:])
                sections: List[Tuple[str, List[str]]] = []
                if prepared:
                    sections.append(("prepared_remarks", prepared))
                if qa:
                    sections.append(("qa", qa))
                return sections

        return [("transcript", list(sentences))]

    # -- stage 3: salience ------------------------------------------------------
    def score_sentence(self, sentence: str) -> SentenceRecord:
        """Score one sentence for financial salience.

        Numeric density (revenue/margin/guidance figures) carries the substance of an
        earnings call, LM sentiment density marks evaluative language, and hedging
        categories (litigious/constraining/uncertainty) are penalized because they are
        overwhelmingly disclaimer and cover-your-backs language.
        """
        stats = self.lm_scorer.tag_stats(sentence)
        tokens = max(1, stats["tokens"])
        densities = stats["densities"]

        numeric_hits = len(_NUMERIC_TOKEN.findall(sentence))
        numeric_score = min(1.0, numeric_hits / 6.0)

        sentiment = densities.get("Positive", 0.0) + densities.get("Negative", 0.0)
        sentiment_score = min(1.0, 8.0 * sentiment)

        hedging = (
            densities.get("Litigious", 0.0)
            + densities.get("Constraining", 0.0)
            + 0.5 * densities.get("Uncertainty", 0.0)
        )
        hedging_score = min(1.0, 6.0 * hedging)

        salience = (
            self.config.numeric_weight * numeric_score
            + self.config.sentiment_weight * sentiment_score
            - self.config.hedging_weight * hedging_score
        )

        if stats["tokens"] < 6:
            salience *= 0.5

        return SentenceRecord(
            text=sentence,
            section="",
            salience=salience,
            tokens=tokens,
            numeric_hits=numeric_hits,
            lm_counts=dict(stats["counts"]),
        )

    # -- stage 4: chunk + select ------------------------------------------------
    def _chunk_section(
        self, section_name: str, sentences: Sequence[str], chunk_tokens: int
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        current: List[SentenceRecord] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            ranked = sorted((s.salience for s in current), reverse=True)[:3]
            chunks.append(
                ChunkRecord(
                    index=len(chunks),
                    section=section_name,
                    sentences=current,
                    tokens=current_tokens,
                    salience=sum(ranked) / len(ranked),
                )
            )
            current = []
            current_tokens = 0

        for sentence in sentences:
            record = self.score_sentence(sentence)
            record.section = section_name
            if current and current_tokens + record.tokens > chunk_tokens:
                flush()
            current.append(record)
            current_tokens += record.tokens

        flush()
        return chunks

    def plan(self, text: str) -> Dict[str, Any]:
        """Run the non-generative stages and return the plan plus diagnostics.

        The harness and tests use this to inspect selection behavior without paying for
        generation.
        """
        sentences = _split_sentences(text)
        kept, removed = self.strip_boilerplate(sentences)
        sections = self.split_sections(kept)

        chunk_tokens = max(64, min(self.config.chunk_tokens, self._input_limit() - 8))
        chunks: List[ChunkRecord] = []
        for section_name, section_sentences in sections:
            section_chunks = self._chunk_section(section_name, section_sentences, chunk_tokens)
            for chunk in section_chunks:
                chunk.index = len(chunks)
                chunks.append(chunk)

        selected = self._select_chunks(chunks)

        return {
            "sentences": sentences,
            "kept_sentences": kept,
            "removed_sentences": removed,
            "sections": sections,
            "chunks": chunks,
            "selected_chunks": selected,
            "chunk_tokens": chunk_tokens,
        }

    def _section_allowances(
        self, sections: Dict[str, List[ChunkRecord]], budget: int
    ) -> Dict[str, int]:
        """Split the input budget across sections.

        Prepared remarks are capped at ``prepared_remarks_share`` of the budget rather
        than at their proportional size, because the Q&A session is usually several
        times longer while the prepared remarks carry the actual results and guidance.
        Whatever the prepared remarks do not use flows to the Q&A.
        """
        names = list(sections)
        if len(names) == 1:
            return {names[0]: budget}

        allowances: Dict[str, int] = {}
        prepared_name = "prepared_remarks" if "prepared_remarks" in sections else None

        if prepared_name is not None:
            prepared_tokens = sum(chunk.tokens for chunk in sections[prepared_name])
            prepared_budget = int(min(prepared_tokens, budget * self.config.prepared_remarks_share))
            allowances[prepared_name] = prepared_budget
            remainder = budget - prepared_budget
        else:
            remainder = budget

        other_names = [name for name in names if name != prepared_name]
        other_tokens = sum(
            chunk.tokens for name in other_names for chunk in sections[name]
        ) or 1
        for name in other_names:
            share = sum(chunk.tokens for chunk in sections[name]) / other_tokens
            allowances[name] = int(remainder * share)

        return allowances

    def _select_chunks(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        if not chunks:
            return []

        budget = max(64, self.config.max_input_tokens)

        # The legacy arm must keep reproducing plain truncation.
        if not self.config.select_by_salience:
            selected: List[ChunkRecord] = []
            used_tokens = 0
            for chunk in chunks:
                if selected and used_tokens + chunk.tokens > budget:
                    break
                selected.append(chunk)
                used_tokens += chunk.tokens
                if used_tokens >= budget:
                    break
            return selected

        sections: Dict[str, List[ChunkRecord]] = {}
        for chunk in chunks:
            sections.setdefault(chunk.section, []).append(chunk)

        allowances = self._section_allowances(sections, budget)
        selected = []
        taken: set = set()
        used_tokens = 0

        for section_name, section_chunks in sections.items():
            allowance = allowances.get(section_name, budget)
            section_used = 0
            for chunk in sorted(section_chunks, key=lambda c: c.salience, reverse=True):
                if section_used and section_used + chunk.tokens > allowance:
                    continue
                selected.append(chunk)
                taken.add(id(chunk))
                section_used += chunk.tokens
                used_tokens += chunk.tokens

        # Spend anything left over on the strongest remaining chunks, so a short
        # prepared-remarks section never leaves budget on the table.
        remaining = budget - used_tokens
        if remaining > 0:
            for chunk in sorted(chunks, key=lambda c: c.salience, reverse=True):
                if id(chunk) in taken or chunk.tokens > remaining:
                    continue
                selected.append(chunk)
                taken.add(id(chunk))
                remaining -= chunk.tokens

        selected.sort(key=lambda chunk: chunk.index)
        return selected

    # -- stage 5: generation ----------------------------------------------------
    def _is_noise(self, sentence: str) -> bool:
        """True for boilerplate/logistics sentences (used to clean generated summaries)."""
        if any(pattern.search(sentence) for pattern in _STRONG_BOILERPLATE):
            return True
        if sum(1 for pattern in _WEAK_BOILERPLATE if pattern.search(sentence)) >= 2:
            return True
        return any(pattern.search(sentence) for pattern in _PREAMBLE_DISCOURSE)

    def _select_summary_sentences(
        self, chunk_summaries: Sequence[str], per_chunk: Optional[int] = None
    ) -> str:
        """Pick the strongest sentences from every chunk summary, preserving breadth.

        BART is trained on news leads, so feeding it the concatenated chunk summaries
        makes it re-summarize the opening topic and drop everything else. Selecting a
        few salient sentences per chunk first keeps every chunk represented, and the
        synthesis pass then merges them into prose.
        """
        per_chunk = per_chunk or self.config.synthesis_sentences_per_chunk
        picked: List[str] = []
        seen = set()

        for summary in chunk_summaries:
            sentences = [
                sentence
                for sentence in _split_sentences(summary)
                if len(sentence.split()) >= 5 and not self._is_noise(sentence)
            ]
            ranked = sorted(
                sentences, key=lambda sentence: self.score_sentence(sentence).salience, reverse=True
            )
            for sentence in ranked[: max(1, per_chunk)]:
                key = re.sub(r"[^a-z0-9 ]", "", sentence.lower())[:80]
                if key in seen:
                    continue
                seen.add(key)
                picked.append(sentence)

        return " ".join(picked)

    def _synthesize(
        self,
        chunk_summaries: Sequence[str],
        sections: Sequence[str],
        max_tokens: int,
        min_tokens: int,
    ) -> Tuple[str, str]:
        """Combine chunk summaries into the final executive summary.

        A single global reduce collapses onto whichever topic comes first (BART is
        trained on news leads), so when prepared remarks and Q&A are both present each
        is reduced on its own and the two are labelled. Returns ``(summary, mode)``.
        """
        if not chunk_summaries:
            return "", "empty"

        grouped: List[Tuple[str, List[str]]] = []
        for summary, section in zip(chunk_summaries, sections):
            if grouped and grouped[-1][0] == section:
                grouped[-1][1].append(summary)
            else:
                grouped.append((section, [summary]))

        if len(grouped) <= 1:
            name = grouped[0][0] if grouped else "transcript"
            group = grouped[0][1] if grouped else []
            if len(group) == 1:
                return group[0], "single_chunk"
            seeded = self._select_summary_sentences(group)
            if not seeded:
                seeded = " ".join(group)
            return (
                self._generate(seeded, max_new_tokens=max_tokens, min_new_tokens=min_tokens),
                "map_reduce",
            )

        parts: List[str] = []
        for section_name, group in grouped:
            if len(group) == 1:
                text = group[0]
            else:
                seeded = self._select_summary_sentences(group)
                if not seeded:
                    seeded = " ".join(group)
                # The Q&A gets a tighter budget so the two sections stay comparable in
                # length even though the prepared remarks are usually far longer.
                scale = 1.0 if section_name == "prepared_remarks" else 0.7
                text = self._generate(
                    seeded,
                    max_new_tokens=max(48, int(max_tokens * scale)),
                    min_new_tokens=max(24, int(min_tokens * scale)),
                )
            label = _SECTION_LABELS.get(section_name, section_name.replace("_", " ").title())
            parts.append(f"{label}: {text.strip()}")

        return "\n\n".join(parts), "sectioned"

    def _generate(self, text: str, max_new_tokens: int, min_new_tokens: int) -> str:
        model = self._ensure_model()
        tokenizer = self._ensure_tokenizer()

        inputs = tokenizer(
            text,
            max_length=self._input_limit(),
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_new_tokens,
                min_length=min_new_tokens,
                num_beams=self.config.num_beams,
                early_stopping=True,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Return just the summary string (compatibility entry point)."""
        return self.summarize_with_metadata(text, max_length=max_length, min_length=min_length)[
            "summary"
        ]

    def summarize_with_metadata(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize ``text`` and return the summary plus selection diagnostics."""
        max_summary_tokens = max_length or self.config.max_summary_tokens
        min_summary_tokens = min_length or self.config.min_summary_tokens

        if not text or not text.strip():
            logger.warning("Empty text provided for summarization.")
            return self._empty_result("No text available to summarize.")

        if len(text.split()) < 30:
            logger.warning("Text too short for summarization (%d words)", len(text.split()))
            return self._empty_result("Text too short to generate an abstractive summary.")

        plan = self.plan(text)
        selected: List[ChunkRecord] = plan["selected_chunks"]

        if not selected:
            logger.warning("No summarizable content survived preprocessing.")
            return self._empty_result("Unable to generate summary from provided text.")

        # An explicit max_length/min_length override also caps the per-chunk pass,
        # so callers asking for a short summary actually get one.
        chunk_max_tokens = min(self.config.chunk_summary_tokens, max_summary_tokens)
        chunk_min_tokens = min(self.config.chunk_summary_min_tokens, min_summary_tokens)

        chunk_summaries: List[str] = []
        for position, chunk in enumerate(selected, start=1):
            logger.debug(
                "Summarizing chunk %d/%d (section=%s, tokens=%d, salience=%.3f)",
                position,
                len(selected),
                chunk.section,
                chunk.tokens,
                chunk.salience,
            )
            chunk_summaries.append(
                self._generate(
                    chunk.text,
                    max_new_tokens=chunk_max_tokens,
                    min_new_tokens=chunk_min_tokens,
                )
            )

        if len(chunk_summaries) == 1:
            summary = chunk_summaries[0]
            synthesis_mode = "single_chunk"
        elif not self.config.select_by_salience:
            # Legacy arm: one plain generation over the selected (truncated) text.
            summary = self._generate(
                " ".join(chunk.text for chunk in selected),
                max_new_tokens=max_summary_tokens,
                min_new_tokens=min_summary_tokens,
            )
            synthesis_mode = "truncation"
        else:
            summary, synthesis_mode = self._synthesize(
                chunk_summaries,
                [chunk.section for chunk in selected],
                max_tokens=max_summary_tokens,
                min_tokens=min_summary_tokens,
            )

        if not summary or not summary.strip():
            logger.warning("Summary generation produced empty result")
            summary = "Unable to generate summary from provided text."

        # BART occasionally decodes "...quarter.Operator" with no space; repair the join.
        summary = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", summary)

        transcript_tokens = self._token_len(text)
        summary_tokens = self._token_len(summary)

        sections_meta = []
        for section_name, section_sentences in plan["sections"]:
            section_chunks = [c for c in plan["chunks"] if c.section == section_name]
            chosen = [c for c in selected if c.section == section_name]
            sections_meta.append(
                {
                    "name": section_name,
                    "sentences": len(section_sentences),
                    "chunks": len(section_chunks),
                    "selected_chunks": len(chosen),
                    "selected_tokens": sum(c.tokens for c in chosen),
                }
            )

        return {
            "summary": summary,
            "transcript_tokens": transcript_tokens,
            "summary_tokens": summary_tokens,
            "compression_ratio": (
                round(summary_tokens / transcript_tokens, 4) if transcript_tokens else 0.0
            ),
            "sentences_total": len(plan["sentences"]),
            "sentences_after_boilerplate": len(plan["kept_sentences"]),
            "boilerplate_removed_count": len(plan["removed_sentences"]),
            "boilerplate_removed_samples": plan["removed_sentences"][:5],
            "chunks_total": len(plan["chunks"]),
            "chunks_used": len(selected),
            "input_tokens": sum(chunk.tokens for chunk in selected),
            "sections": sections_meta,
            "chunk_summaries": chunk_summaries,
            "synthesis_mode": synthesis_mode,
            "used_boilerplate_strip": self.config.strip_boilerplate,
            "used_salience_selection": self.config.select_by_salience,
            "model": self.config.model_name,
        }

    def _empty_result(self, message: str) -> Dict[str, Any]:
        return {
            "summary": message,
            "transcript_tokens": 0,
            "summary_tokens": 0,
            "compression_ratio": 0.0,
            "sentences_total": 0,
            "sentences_after_boilerplate": 0,
            "boilerplate_removed_count": 0,
            "boilerplate_removed_samples": [],
            "chunks_total": 0,
            "chunks_used": 0,
            "input_tokens": 0,
            "sections": [],
            "chunk_summaries": [],
            "synthesis_mode": "empty",
            "used_boilerplate_strip": self.config.strip_boilerplate,
            "used_salience_selection": self.config.select_by_salience,
            "model": self.config.model_name,
        }


def _split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences without pulling in SpaCy."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def legacy_summarizer_config(model_name: str = DEFAULT_MODEL_NAME) -> SummarizerConfig:
    """Config reproducing the old 'truncate to 1024 tokens, summarize' behavior."""
    return SummarizerConfig(
        model_name=model_name,
        chunk_tokens=1024,
        max_input_tokens=1024,
        strip_boilerplate=False,
        select_by_salience=False,
    )
