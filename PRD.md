# PRD: Khmer Word-Level Next-Word Prediction

## 1. Overview

This project builds a next-word prediction (autocomplete) system for the Khmer language, modeled on smartphone keyboard suggestion features common in high-resource languages. The system takes a sequence of typed Khmer words and predicts the most likely next word(s). Core deliverable is an LSTM-based language model trained on segmented Khmer Wikipedia text; fine-tuning a pretrained multilingual transformer is a stretch goal.

## 2. Goals & Non-Goals

**Goals:**
- Segment raw, unsegmented Khmer text into words using an existing published tool
- Train a working LSTM language model for next-word prediction
- Evaluate the model quantitatively (perplexity, top-k accuracy) and validate segmentation quality against a gold-standard benchmark
- Produce a live, interactive demo
- (Stretch) Fine-tune a small transformer and compare against the LSTM

**Non-Goals:**
- Not building a production-ready keyboard/IME application
- Not building a segmentation model from scratch (using existing tools instead)
- Not handling code-mixed Khmer-English or heavily informal/social-media text
- Not supporting dialectal variation or non-standard Khmer script variants
- Not optimizing for inference latency/deployment at scale

## 3. Users & Use Case

Primary hypothetical user: a Khmer speaker typing on a phone or computer without access to predictive text support, who would benefit from faster, more accurate typing. Academic framing: this project targets a genuine, real-world gap rather than a purely synthetic benchmark exercise.

## 4. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | System accepts a Khmer text prefix (sequence of words) as input |
| FR2 | System returns the top-k (k=3–5) most probable next words for a given prefix |
| FR3 | System handles out-of-vocabulary/rare words gracefully via `<UNK>` token, without crashing |
| FR4 | Preprocessing pipeline converts raw unsegmented Khmer text into word-segmented, vocabulary-mapped training sequences |
| FR5 | Demo interface allows a user to type Khmer text and see live suggestions update |
| FR6 (stretch) | System supports switching between LSTM and fine-tuned transformer for comparison |

## 5. Technical Architecture

```
Raw Khmer Wikipedia text
        │
        ▼
  Cleaning (Unicode normalization, markup/non-Khmer stripping)
        │
        ▼
  Word Segmentation (khmer-nltk / khmercut)
        │
        ▼
  Vocabulary Building (frequency cutoff, <UNK> handling)
        │
        ▼
  Sequence Generation (context window → next-word pairs)
        │
        ▼
  ┌─────────────────┐      ┌──────────────────────────┐
  │  LSTM Model      │      │ (Stretch) Fine-tuned      │
  │  (core)          │      │ Transformer (XLM-R-base)  │
  └─────────────────┘      └──────────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                        ▼
              Evaluation (perplexity, top-k accuracy)
                        │
                        ▼
              Demo Interface (Gradio/Streamlit)
```

**Key tools:** Python, PyTorch, HuggingFace `datasets` and `transformers`, `khmer-nltk`/`khmercut`, Google Colab (GPU), Gradio or Streamlit for demo.

## 6. Data Requirements

- **Primary:** Khmer Wikipedia (`wikimedia/wikipedia`, `km` subset), ~12,000 articles, HuggingFace-hosted, open license
- **Backup/supplement:** CC-100 Khmer subset, if Wikipedia proves insufficient in volume
- **Validation reference:** khPOS corpus (Ye Kyaw Thu et al., 2017) — 12,000 manually segmented, POS-tagged sentences, used to benchmark our segmentation pipeline's precision/recall

## 7. Milestones & Timeline

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Setup & Learning | 1–2 | Environment ready, tools tested on sample sentences |
| Data Acquisition & Exploration | 2–3 | Exploratory notebook with corpus stats |
| Preprocessing Pipeline | 3–5 | Clean, reusable preprocessing script; segmentation spot-checked |
| Baseline LSTM | 5–8 | Trained LSTM checkpoint with decreasing validation loss |
| Evaluation | 8–9 | Perplexity, top-k accuracy, segmentation precision/recall reported |
| Transformer Fine-tuning (stretch) | 9–11 | Fine-tuned model + comparison to LSTM |
| Demo Interface | 11–12 | Working interactive demo |
| Write-up & Presentation | 12–14 | Final report, rehearsed live demo |

## 8. Success Metrics / Acceptance Criteria

| Metric | Target |
|--------|--------|
| Perplexity (LSTM, held-out validation) | Lower than a naive unigram-frequency baseline (establish baseline number during Phase 4) |
| Top-3 accuracy | ≥ 25–30% |
| Top-5 accuracy | ≥ 35–40% |
| Segmentation precision/recall vs. khPOS | ≥ 90% (in line with published tool benchmarks) |
| Demo | Successfully returns live suggestions for arbitrary typed Khmer input, in-presentation |
| (Stretch) Transformer vs. LSTM | Transformer shows measurable improvement in perplexity or top-k accuracy over LSTM |

*Note: these are reasonable starting targets, not hard guarantees — treat Phase 4 as the point to sanity-check and, if needed, revise them based on what your actual corpus size and baseline numbers support.*

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Segmentation errors propagate into model | Spot-check against khPOS early; document error rate as a known limitation |
| Wikipedia corpus too small for stable LSTM training | Supplement with CC-100 Khmer; fall back to smaller model/embedding size if needed |
| Colab GPU time limits/session drops | Checkpoint model regularly; save to Google Drive; budget buffer time |
| Transformer fine-tuning doesn't finish in time | Explicitly scoped as stretch goal; LSTM-only is an acceptable, complete outcome |
| Team falls behind schedule | Phase 4 (Evaluation) is the checkpoint to cut stretch goals and prioritize polishing core deliverable + demo |

## 10. Implementation Checklist

### Phase 0: Setup & Learning (Week 1–2)
- [ ] Set up Colab notebook / local Jupyter environment
- [ ] Create shared GitHub repo for team
- [ ] Install core libraries (`torch`, `datasets`, `transformers`, `khmer-nltk`)
- [ ] Go through a basic PyTorch tutorial (tensors, training loop, `nn.Module`)
- [ ] Test `khmer-nltk`/`khmercut` on a few sample Khmer sentences
- [ ] Confirm Colab GPU access works (`torch.cuda.is_available()`)

### Phase 1: Data Acquisition & Exploration (Week 2–3)
- [ ] Load `wikimedia/wikipedia` (`km` subset) via HuggingFace `datasets`
- [ ] Count total articles, words, characters
- [ ] Check article length distribution (spot stub articles)
- [ ] Spot-check a handful of raw articles for content quality
- [ ] Write short exploratory notebook summarizing findings

### Phase 2: Preprocessing Pipeline (Week 3–5)
- [ ] Write Unicode normalization function (NFC)
- [ ] Strip wiki markup, HTML remnants, non-Khmer characters
- [ ] Test segmentation on small subset (~100 articles) before scaling up
- [ ] Run segmentation on full corpus; cache results to disk
- [ ] Manually review ~50–100 segmented sentences for quality
- [ ] Compare segmentation output against khPOS (precision/recall)
- [ ] Build vocabulary (frequency cutoff, `<UNK>` handling)
- [ ] Generate training sequences (context window → next-word pairs)
- [ ] Document each preprocessing step for the final report

### Phase 3: Baseline LSTM Model (Week 5–8)
- [ ] Implement `Dataset`/`DataLoader` for training sequences
- [ ] Implement LSTM model (embedding → LSTM → linear/softmax)
- [ ] Split data into train/validation sets
- [ ] Write training loop with loss tracking
- [ ] Run first training pass; debug shape/encoding issues
- [ ] Tune hyperparameters (hidden size, layers, learning rate)
- [ ] Save trained model checkpoint

### Phase 4: Evaluation (Week 8–9)
- [ ] Implement perplexity calculation on validation set
- [ ] Implement top-k accuracy calculation
- [ ] Establish naive baseline (e.g., unigram frequency) for comparison
- [ ] Record segmentation precision/recall vs. khPOS
- [ ] Generate qualitative example predictions
- [ ] **Checkpoint: decide whether to pursue stretch goal based on progress**

### Phase 5: Transformer Fine-tuning — Stretch (Week 9–11)
- [ ] Select pretrained multilingual model (e.g., XLM-R-base)
- [ ] Prepare data pipeline for fine-tuning
- [ ] Fine-tune on Colab GPU
- [ ] Evaluate using same metrics as LSTM
- [ ] Compare results side-by-side with LSTM

### Phase 6: Demo Interface (Week 11–12)
- [ ] Build simple Gradio/Streamlit interface
- [ ] Wire up live inference (typed prefix → top-k suggestions)
- [ ] Test demo with various inputs, including edge cases (empty input, rare words)
- [ ] Polish UI (labels, instructions)

### Phase 7: Write-up & Presentation (Week 12–14)
- [ ] Draft preprocessing section (with examples)
- [ ] Draft model architecture & training section
- [ ] Draft evaluation section (tables/charts)
- [ ] Draft limitations & future work section
- [ ] Assemble final report
- [ ] Prepare slides/presentation
- [ ] Rehearse live demo (have backup examples ready)

## 11. Roles & Responsibilities

| Area | Owner |
|------|-------|
| Data acquisition & preprocessing pipeline | [Name] |
| Segmentation validation (khPOS comparison) | [Name] |
| LSTM model implementation & training | [Name] |
| Evaluation (metrics, benchmarking) | [Name] |
| Transformer fine-tuning (stretch) | [Name] |
| Demo interface | [Name] |
| Final report & presentation | [Name] (or shared) |
