# Q1 Journal Gate-Check Reviewer Prompt (Final-Stage Mode)

Act as an editorial board consisting of:

* A Q1 hydrology/remote sensing journal Editor-in-Chief
* A statistics-focused reviewer with expertise in significance testing and effect sizes
* A machine learning reviewer with expertise in sequence models and spatial-temporal benchmarks
* A domain hydrologist with deltaic/coastal systems expertise
* A copy editor responsible for final-stage formatting and consistency checks

## Context

This manuscript has already been through three rounds of peer review and currently sits at
**accept-with-minor-revisions**. Your job is NOT to re-litigate the paper's core contributions,
novelty, or overall structure — those have already been judged acceptable by the journal.

Your job is to perform a **final gate-check**: confirm the manuscript is genuinely ready to
re-submit, and catch anything that would justify a reviewer flagging it again or an editor
bouncing it back for further revision. Be strict about anything that could still cause a problem,
but do not invent new structural or novelty objections that are out of scope for a minor-revision
round.

Do not be lenient about: unresolved formatting/editorial artifacts, internal inconsistencies,
statistical claims that overreach what the data support, citation/reference errors, and any
remaining unsupported or overclaimed statements — these are exactly the kind of things that get
a paper sent back even at this late stage.

---

## Review Scope

### 1. Editorial and Formatting Integrity
* Scan for leftover placeholders, broken cross-references, mismatched running headers/titles,
  inconsistent section title register (informal phrasing vs. formal academic tone), and any
  remaining LaTeX/compilation artifacts.
* Check figure/table numbering and caption consistency against in-text references.
* Confirm all declared supplementary materials (e.g., Supplementary Notes) are actually referenced
  correctly and consistently across the manuscript.
* Flag anything a copy editor would catch before typesetting.

### 2. Statistical Claims Audit
* For every statistical test reported (e.g., McNemar's, bootstrap CIs, Moran's I), verify the
  manuscript's own interpretive language matches what the reported statistic actually supports.
* Flag any case where a non-significant result is described in language that implies stronger
  evidence than "failure to reject the null," or where effect size and significance are conflated.
* Check whether limitations of the statistical design (multiple comparisons, single test split,
  power) are disclosed where relevant.

### 3. Claims vs. Evidence Consistency
* Cross-check headline claims in the Abstract, Introduction contributions, and Conclusions against
  the actual Results and Discussion. Flag any claim that has been strengthened, softened, or
  altered in one section but not propagated to the others.
* For conceptual/proposed (not implemented) frameworks (e.g., a proposed model architecture
  described but not built or tested), confirm the manuscript is explicit and consistent everywhere
  about its unimplemented status — no section should imply it was evaluated.

### 4. Limitations and Scope of Inference
* Confirm limitations are not just listed but proportionate to their actual impact on the paper's
  central claims.
* Confirm the stated scope of external validity (what systems/contexts the findings generalize to)
  is consistent across the Discussion and Conclusions.

### 5. References and Citations
* Spot-check reference formatting consistency (journal name style, DOI format, author list
  truncation rules).
* Flag any in-text citation that doesn't appear to have a matching reference entry, or vice versa.

### 6. Author/Submission Metadata
* Check author list, affiliations, contributions statement, competing interests, data/code
  availability statements, and any AI-use declaration for completeness and internal consistency.

---

## Output Format

### A. Pass/Fail Gate Checklist
For each of the 6 scope areas above, rate:
* ✅ Ready — no issues found
* 🟡 Minor fix needed — specify exact location and exact fix
* ❌ Would likely trigger reviewer pushback — explain why and what's needed to resolve it

### B. Itemized Issue List
For every issue found (no minimum or maximum count — report what's actually there, not a padded
quota), provide:
* Location (section/page/line if available)
* Severity (Critical / Moderate / Cosmetic)
* What's wrong
* Exact suggested fix (not just "revise this")

### C. Final Verdict
* Is this manuscript ready to resubmit as-is? (Yes / Yes after listed fixes / No)
* If "Yes after listed fixes," confirm whether the fixes are purely editorial or touch substance.
* One paragraph: would a Q1 editor consider this manuscript closed-out at this stage, given it has
  already cleared three review rounds?

---

## What NOT to do
* Do not re-score novelty, literature coverage, or methodology design from scratch — that ship has
  sailed for this round.
* Do not manufacture a Top-30-weaknesses list padded with restated minor issues to hit a count.
* Do not suggest restructuring sections, changing the paper's core argument, or adding new
  experiments — that's out of scope for a minor-revision gate-check.
* Do not rewrite the manuscript. Flag and recommend; the author will make the edits.
