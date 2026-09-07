---
name: model-scanner
description: Inspect a pinned model revision and produce an evidence-backed ModelSupportCard for vLLM-Kunlun adaptation.
---

# Model Scanner Skill

## Purpose

Build a structured model inventory before any source patching. The inventory must describe configuration, architecture, attention, MoE, quantization, multimodal components, dynamic-shape risks, and likely runtime paths.

## Rules

Read only the pinned model revision from the Task Contract. Do not modify the target repository, change model files, relax acceptance thresholds, or claim runtime support from a static scan alone.

## Required tools

1. `inspect_model_config` to load configuration and revision metadata.
2. `scan_model_modules` to enumerate model modules and relevant symbols.
3. `package_support_card` to produce JSON and Markdown artifacts.

## Acceptance

The Skill is complete only when all required fields in MAT-002 are populated, evidence paths are recorded, and unresolved assumptions are listed as limitations. The next Task is responsible for capability matching and gap classification.
