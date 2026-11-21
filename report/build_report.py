"""Builds report/RL_Summative_Report.docx from results in logs/ and figures in assets/.

    uv run --with python-docx report/build_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ROOT / "assets"
OUT = ROOT / "report" / "RL_Summative_Report.docx"

STUDENT = "J. Burabyo"            # edit: full name
VIDEO = "[add video link]"         # edit
REPO = "https://github.com/Burabyo/jburabyo_rl_summative"  # edit

df = pd.read_csv(ROOT / "logs" / "sweep_results.csv")
df0 = df[df.seed == 0].copy()
seeds = pd.read_csv(ROOT / "logs" / "seed_robustness.csv")
best = json.loads((ROOT / "models" / "best.json").read_text())
B = {a: df0.set_index("run").loc[v["run"]] for a, v in best["per_algo"].items()}
NAR = json.loads((ROOT / "report" / "narrative.json").read_text())

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.0))
sec.top_margin = sec.bottom_margin = Cm(1.9)

st = doc.styles
st["Normal"].font.name = "Calibri"
st["Normal"].font.size = Pt(11)
st["Normal"].paragraph_format.space_after = Pt(4)
st["Normal"].paragraph_format.line_spacing = 1.0
for lvl, size in ((1, 14), (2, 12), (3, 11)):
    h = st[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.paragraph_format.space_before = Pt(10 if lvl == 1 else 6)
    h.paragraph_format.space_after = Pt(3)
    h.paragraph_format.keep_with_next = True
    rpr = h.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), "Calibri")


def P(text: str, bold_lead: str | None = None, size: float | None = None, italic=False, align=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        if size:
            r.font.size = Pt(size)
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullet(text: str, lead: str | None = None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(1)
    return p


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:color"), "auto")
    s.set(qn("w:fill"), hex_fill)
    tcPr.append(s)


def table(header, rows, widths_cm, font=8.5, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(font)
        shade(c, "DCE6F2")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
            if highlight_row is not None and ri == highlight_row:
                r.bold = True
                shade(cells[i], "E8F5E9")
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path: Path, caption: str, width_cm=16.0):
    doc.add_picture(str(path), width=Cm(width_cm * 0.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = P(caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    c.paragraph_format.space_after = Pt(6)


# ------------------------------------------------------------------ title block
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Reinforcement Learning Summative Assignment Report")
r.bold, r.font.size = True, Pt(16)
t2 = doc.add_paragraph()
t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t2.add_run("Imisoro Compliance Coach: learning when to remind, teach and escalate for Rwandan small-business tax compliance")
r.italic, r.font.size = True, Pt(11)
P(STUDENT, "Student Name: ")
P(VIDEO, "Video Recording: ")
P(REPO, "GitHub Repository: ")

# ------------------------------------------------------------------ 1
doc.add_heading("1. Project Overview", 1)
P(NAR["overview"])

# ------------------------------------------------------------------ 2
doc.add_heading("2. Environment Description", 1)
doc.add_heading("2.1 Agent(s)", 2)
P(NAR["agent"])
P(NAR["why_rl"], "Why RL is needed. ")

doc.add_heading("2.2 Action Space", 2)
P("Discrete(8). One intervention per week. Each action maps to a feature of the Imisoro product or "
  "to an existing RRA channel:")
acts = [
    ("0 WAIT", "No contact; fatigue decays.", "0"),
    ("1 SMS_REMINDER", "Deadline SMS. Raises salience and preparation only in the deadline week or the week before.", "−0.10"),
    ("2 EMAIL_DIGEST", "Email with personalised tax calendar; response depends on digital literacy.", "−0.05"),
    ("3 RUN_CALCULATOR", "Prompt to compute regime/VAT/PAYE liability; preparation gain scales with record quality.", "−0.20"),
    ("4 EDUCATION_MODULE", "Bilingual micro-lesson + quiz (RAG answer); raises tax knowledge permanently.", "−0.30"),
    ("5 RECORDS_NUDGE", "Prompt to log sales / reconcile EBM receipts; raises records readiness.", "−0.20"),
    ("6 SAVINGS_NUDGE", "Prompt to set aside the calculated tax; raises cash ratio for the next payment.", "−0.10"),
    ("7 ESCALATE_TO_RRA", "Book an RRA call-centre/office session; large preparation gain, max 3 per year.", "−2.00"),
]
table(["Action", "Real-world meaning and effect", "Cost"], acts, [3.6, 11.0, 1.6])

doc.add_heading("2.3 Observation Space", 2)
P("Box(0, 1, shape=(21,)). The owner's true knowledge, trust and fatigue are hidden (partially observable). "
  "The agent sees only signals the app can actually collect:")
obs = [
    ("week_frac, weeks_to_deadline, n_due_next", "w/52; min(Δw,8)/8; count/3", "Tax-calendar engine built from profile", "Weekly"),
    ("next_vat / paye / quarterly / annual", "one-hot", "Calendar engine (RRA due dates)", "Weekly"),
    ("regime one-hot, has_employees, vat_registered", "binary", "Onboarding profile + turnover calculator", "At onboarding"),
    ("digital_literacy", "[0,1]", "Onboarding survey (channel preference)", "At onboarding"),
    ("prep_progress", "[0,1]", "Calculator/draft-return completion in app", "On app event"),
    ("records_score", "[0,1]", "EBM sync / sales-log days in last 30", "Daily sync"),
    ("cash_ratio", "set-aside ÷ liability", "In-app savings tracker (self-report)", "On update"),
    ("engagement_rate", "EMA of responses", "SMS/email open & reply logs", "On contact"),
    ("contacts_4w", "count/4", "Message log", "Weekly"),
    ("quiz_score", "knowledge + N(0,0.05)", "Quiz after lessons (noisy proxy)", "On lesson"),
    ("missed_frac, escalations_left", "count/5; left/3", "Filing confirmation (e-Tax), escalation log", "Per deadline"),
]
table(["Observation(s)", "Encoding / range", "Source or simulation rule", "Cadence"], obs, [5.0, 3.3, 5.6, 2.3], font=8)
P("All features are clipped to [0,1] and fed directly into the MLP input layer as a float32 vector "
  "(agent-input path: env → TorchRL GymWrapper → 'observation' key → network).", size=10)

doc.add_heading("2.4 Reward Structure", 2)
P(NAR["reward"])
P("r_t = −c(a_t) + Σ_obligations due at t [ 10·1(on-time ∧ cash ≥ 0.6) + 4·1(on-time ∧ cash < 0.6) − 15·1(missed) ] "
  "− 25·1(opt-out) + 5·knowledge·1(t = 52)", italic=True, size=10)
P(NAR["p_on_time"], size=10)

doc.add_heading("2.5 Start State and Terminal Conditions", 2)
P(NAR["start_terminal"])

doc.add_heading("2.6 Environment Visualization", 2)
figure(A / "env_render.png", "Figure 1. Three.js 3D view: the small business (centre), Imisoro AI tower, RRA office, "
       "the 52-week calendar ring with obligation pillars (green = filed on time, red = missed), live owner-state bars "
       "and HUD. Message orbs animate each intervention.", 15.5)
P(NAR["visual"], size=10)

# ------------------------------------------------------------------ 3
doc.add_heading("3. System Analysis and Design", 1)
doc.add_heading("3.1 Deep Q-Network (DQN)", 2)
P(NAR["dqn_design"])
doc.add_heading("3.2 Policy Gradient Methods (REINFORCE, PPO, A2C)", 2)
P(NAR["pg_design"])
P(NAR["fairness"], "Fair comparison. ")

# ------------------------------------------------------------------ 4
doc.add_heading("4. Implementation (Hyperparameter Experiments)", 1)
P(NAR["impl_intro"])

def fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


SPECS = {
    "dqn": ("4.1 DQN", ["#", "LR", "γ", "Buffer", "Batch", "Exploration (ε end / decay frac)", "Target update", "Double", "Upd/batch", "Hidden",
                         "Mean reward", "On-time %"],
            lambda r: [r.idx, fmt(r.hp_lr), fmt(r.hp_gamma), int(r.hp_buffer_size), int(r.hp_batch_size),
                       f"{fmt(r.hp_eps_end)} / {fmt(r.hp_eps_fraction)}",
                       f"soft τ={fmt(r.hp_tau)}" if r.hp_target_update == "soft" else f"hard/{int(r.hp_hard_interval)}",
                       "Y" if str(r.hp_double_dqn) == "True" else "N", int(r.hp_updates_per_batch), int(r.hp_hidden),
                       f"{r.eval_return_mean:+.1f}", f"{100 * r.eval_compliance:.1f}"],
            [0.6, 1.1, 1.0, 1.2, 1.0, 2.4, 1.9, 1.0, 1.1, 1.1, 1.6, 1.4]),
    "reinforce": ("4.2 REINFORCE", ["#", "LR", "γ", "Episodes/update", "Baseline", "Return norm.", "Entropy coef", "Hidden",
                                    "Mean reward", "On-time %"],
                  lambda r: [r.idx, fmt(r.hp_lr), fmt(r.hp_gamma), int(r.hp_episodes_per_batch), r.hp_baseline,
                             "Y" if str(r.hp_normalize_returns) == "True" else "N", fmt(r.hp_entropy_coeff),
                             int(r.hp_hidden), f"{r.eval_return_mean:+.1f}", f"{100 * r.eval_compliance:.1f}"],
                  [0.6, 1.3, 1.2, 2.0, 1.8, 1.7, 1.8, 1.3, 2.0, 1.7]),
    "ppo": ("4.3 PPO", ["#", "LR", "γ", "GAE λ", "Clip ε", "Epochs", "Minibatch", "Rollout (envs×steps)", "Entropy coef",
                        "Mean reward", "On-time %"],
            lambda r: [r.idx, fmt(r.hp_lr), fmt(r.hp_gamma), fmt(r.hp_gae_lambda), fmt(r.hp_clip_epsilon),
                       int(r.hp_epochs), int(r.hp_minibatch_size), f"{int(r.hp_n_envs)}×{int(r.hp_steps_per_env)}",
                       fmt(r.hp_entropy_coeff), f"{r.eval_return_mean:+.1f}", f"{100 * r.eval_compliance:.1f}"],
            [0.6, 1.2, 1.1, 1.2, 1.2, 1.2, 1.6, 2.4, 1.7, 1.8, 1.6]),
    "a2c": ("4.4 Actor-Critic (A2C)", ["#", "LR", "γ", "GAE λ", "n-steps × envs", "Entropy coef", "Critic coef", "Adv. norm",
                                       "Hidden", "Mean reward", "On-time %"],
            lambda r: [r.idx, fmt(r.hp_lr), fmt(r.hp_gamma), fmt(r.hp_gae_lambda),
                       f"{int(r.hp_steps_per_env)} × {int(r.hp_n_envs)}", fmt(r.hp_entropy_coeff), fmt(r.hp_critic_coeff),
                       "Y" if str(r.hp_normalize_advantage) == "True" else "N", int(r.hp_hidden),
                       f"{r.eval_return_mean:+.1f}", f"{100 * r.eval_compliance:.1f}"],
            [0.6, 1.2, 1.1, 1.2, 2.0, 1.7, 1.6, 1.5, 1.3, 1.8, 1.6]),
}
for algo, (title, header, rowf, widths) in SPECS.items():
    doc.add_heading(title, 2)
    g = df0[df0.algo == algo].sort_values("idx")
    rows = [rowf(r) for r in g.itertuples()]
    hi = list(g.run).index(best["per_algo"][algo]["run"])
    table(header, rows, widths, font=7.5, highlight_row=hi)
    P(NAR[f"{algo}_analysis"], size=10)

# ------------------------------------------------------------------ 5
doc.add_heading("5. Results Discussion", 1)
doc.add_heading("5.1 Cumulative Rewards", 2)
figure(A / "fig1_reward_comparison.png", "Figure 2. Mean training episode return of the best configuration of each algorithm (rolling mean).", 15)
P(NAR["cumulative"])
summ = []
for a, lab in (("dqn", "DQN"), ("reinforce", "REINFORCE"), ("ppo", "PPO"), ("a2c", "A2C")):
    r = B[a]
    s = seeds[seeds.algo == a]
    vals = [r.eval_return_mean] + s.eval_return_mean.tolist()
    summ.append([lab, best["per_algo"][a]["run"], f"{r.eval_return_mean:+.1f} ± {r.eval_return_std:.1f}",
                 f"{100 * r.eval_compliance:.1f}", f"{r.eval_filed:.1f} / {r.eval_missed:.1f}",
                 f"{pd.Series(vals).mean():+.1f} ± {pd.Series(vals).std(ddof=0):.1f}",
                 f"{r.stress_return_mean:+.1f}", f"{100 * r.stress_compliance:.1f}"])
table(["Algorithm", "Best run", "Eval return (200 businesses)", "On-time %", "Filed / missed per yr",
       "Return over 3 seeds", "Stress return", "Stress on-time %"], summ,
      [2.0, 2.6, 2.6, 1.5, 2.0, 2.4, 1.6, 1.6], font=8)
P(NAR["baselines"], size=10)

doc.add_heading("5.2 Training Stability", 2)
figure(A / "fig3_dqn_objective.png", "Figure 3. DQN objective: TD loss (log scale), mean max-Q on replay batches and ε schedule.", 16)
figure(A / "fig4_pg_entropy.png", "Figure 4. Policy entropy of the policy-gradient methods (left) and PPO's sensitivity to the entropy coefficient (right).", 15.5)
P(NAR["stability"])

doc.add_heading("5.3 Episodes to Converge", 2)
figure(A / "fig2_sweep_convergence.png", "Figure 5. All 12 hyperparameter runs per algorithm; the best run is highlighted.", 15.5)
P(NAR["convergence"])

doc.add_heading("5.4 Generalization", 2)
figure(A / "fig5_generalization.png", "Figure 6. Greedy evaluation on 200 unseen businesses from the training distribution and on the harder, unseen 'stress' population, compared with rule-based policies.", 16)
P(NAR["generalization"])
figure(A / "fig6_action_distribution.png", "Figure 7. Share of weeks in which each best agent selects each intervention.", 14.5)
P(NAR["behaviour"])

# ------------------------------------------------------------------ 6
doc.add_heading("6. Conclusion and Discussion", 1)
P(NAR["conclusion"])
P(NAR["limitations"], "Limitations. ")
P(NAR["integration"], "Integration path. ")
P(NAR["future"], "Future work. ")

doc.save(OUT)
print("wrote", OUT)
