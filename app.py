from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Line fault detector", page_icon="⚡", layout="wide")

# ================================================================ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block-container { max-width: 1150px; padding-top: 2rem; }

.relay {
  border: 1px solid rgba(128,128,128,.35);
  border-radius: 10px;
  padding: 1rem 1.3rem 1.1rem 1.3rem;
  height: 100%;
}
.relay h4 { margin: 0 0 .15rem 0; font-size: 1.05rem; font-weight: 600; }
.relay .sub { margin: 0 0 .7rem 0; font-size: .85rem; opacity: .7; }
.relay .status { display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .7rem 0; }
.relay .lamp { width: 14px; height: 14px; border-radius: 50%; flex: none; }
.relay .word { font-size: 1.9rem; font-weight: 600; line-height: 1; }
.relay.fault { border-color: #D1344B; box-shadow: inset 4px 0 0 #D1344B; }
.relay.fault .lamp { background: #D1344B; box-shadow: 0 0 10px #D1344B; }
.relay.fault .word { color: #D1344B; }
.relay.ok { border-color: #2E9E6B; box-shadow: inset 4px 0 0 #2E9E6B; }
.relay.ok .lamp { background: #2E9E6B; box-shadow: 0 0 10px #2E9E6B; }
.relay.ok .word { color: #2E9E6B; }
.bar { height: 8px; border-radius: 4px; background: rgba(128,128,128,.25); overflow: hidden; margin: .35rem 0 .3rem 0; }
.bar > div { height: 100%; background: #D1344B; }
.small { font-size: .85rem; opacity: .75; }
</style>
""",
    unsafe_allow_html=True,
)

# ================================================================ load saved models
@st.cache_resource
def load_artifacts():
    here = Path(__file__).resolve().parent
    for root in (here.parent, Path(".")):
        folder = root / "results"
        if (folder / "theta.npz").exists():
            pre = np.load(folder / "preprocessing.npz")
            th = np.load(folder / "theta.npz")
            svm = joblib.load(folder / "svm_rbf_best.joblib")
            return dict(pre=pre, theta_R=th["theta_R"], theta_Q=th["theta_Q"], svm=svm, root=root)
    return None


art = load_artifacts()
if art is None:
    st.error(
        "Could not find the saved models. Run the notebook first so that the `results/` folder "
        "contains `theta.npz`, `preprocessing.npz` and `svm_rbf_best.joblib`, and keep "
        "`results/` next to the `app/` folder."
    )
    st.stop()
pre, theta_R, theta_Q, svm = art["pre"], art["theta_R"], art["theta_Q"], art["svm"]

KEYS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
LABELS = {"Ia": "Ia (A)", "Ib": "Ib (A)", "Ic": "Ic (A)", "Va": "Va (pu)", "Vb": "Vb (pu)", "Vc": "Vc (pu)"}
# (low, high, slider step)
BOUNDS = {
    "Ia": (-1000.0, 1000.0, 0.1), "Ib": (-1000.0, 1000.0, 0.1), "Ic": (-1000.0, 1000.0, 0.1),
    "Va": (-2.0, 2.0, 0.001), "Vb": (-2.0, 2.0, 0.001), "Vc": (-2.0, 2.0, 0.001),
}
SWEEP_DEFAULT = {"Ia": 900.0, "Ib": 900.0, "Ic": 900.0, "Va": 1.2, "Vb": 1.2, "Vc": 1.2}


@st.cache_data
def load_dataset(root_str):
    path = Path(root_str) / "data" / "detect_dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).rename(columns={"Output (S)": "fault"})
    df = df[["fault"] + KEYS].reset_index(drop=True)
    # same split as the assignment, only used to say which split a row belongs to
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    split = np.empty(len(df), dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "validation"
    split[idx[n_train + n_val:]] = "test"
    df["split"] = split
    return df


data = load_dataset(str(art["root"]))


# ================================================================ model code (same logic as the notebook)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_batch(raw):
    """raw: (m, 6) array of Ia, Ib, Ic, Va, Vb, Vc. Returns P_R, P_Q, svm_class, svm_score."""
    raw = np.atleast_2d(raw).astype(float)

    # Set R: 6 standardised measurements (+ intercept) for LR-R and the RBF SVM
    r = (raw - pre["mu_R"]) / pre["sd_R"]
    p_R = sigmoid(np.hstack([np.ones((len(r), 1)), r]) @ theta_R)

    # Set Q: squares from RAW values, then standardise all 12, then intercept
    q = np.hstack([raw, raw ** 2])
    q = (q - pre["mu_Q"]) / pre["sd_Q"]
    p_Q = sigmoid(np.hstack([np.ones((len(q), 1)), q]) @ theta_Q)

    svm_class = svm.predict(r).astype(int)
    svm_score = svm.decision_function(r)
    return p_R, p_Q, svm_class, svm_score


# ================================================================ input state (slider and number box stay in sync)
for k in KEYS:
    st.session_state.setdefault(f"val_{k}", 0.0)
    st.session_state.setdefault(f"sl_{k}", 0.0)
    st.session_state.setdefault(f"nb_{k}", 0.0)


def set_inputs(values):
    for k in KEYS:
        lo, hi, _ = BOUNDS[k]
        v = float(np.clip(values[k], lo, hi))
        st.session_state[f"val_{k}"] = v
        st.session_state[f"sl_{k}"] = v
        st.session_state[f"nb_{k}"] = v


def sync_from(k, source):
    v = float(st.session_state[f"{source}_{k}"])
    st.session_state[f"val_{k}"] = v
    st.session_state[f"sl_{k}"] = v
    st.session_state[f"nb_{k}"] = v


EXAMPLES = {
    # a real row from the dataset (label 0)
    "no_fault": dict(Ia=-63.8853, Ib=0.5907, Ic=63.2946, Va=0.1805, Vb=-0.5915, Vc=0.4111),
    # typical-looking fault values (not a dataset row)
    "fault": dict(Ia=800.0, Ib=20.0, Ic=-20.0, Va=0.05, Vb=-0.55, Vc=0.55),
    "zero": {k: 0.0 for k in KEYS},
}

# ================================================================ sidebar: inputs
with st.sidebar:
    st.header("Line measurements")
    st.caption("Drag a slider or type a value. The answer updates straight away.")
    e1, e2 = st.columns(2)
    e1.button("No-fault example", key="ex_nofault", on_click=set_inputs, args=(EXAMPLES["no_fault"],))
    e2.button("Fault example", key="ex_fault", on_click=set_inputs, args=(EXAMPLES["fault"],))
    st.button("Reset to zero", key="ex_zero", on_click=set_inputs, args=(EXAMPLES["zero"],))

    st.markdown("**Currents**")
    for k in ("Ia", "Ib", "Ic"):
        lo, hi, step = BOUNDS[k]
        st.slider(LABELS[k], lo, hi, key=f"sl_{k}", step=step, on_change=sync_from, args=(k, "sl"))
    st.markdown("**Voltages**")
    for k in ("Va", "Vb", "Vc"):
        lo, hi, step = BOUNDS[k]
        st.slider(LABELS[k], lo, hi, key=f"sl_{k}", step=step, on_change=sync_from, args=(k, "sl"))

    with st.expander("Type exact values"):
        for k in KEYS:
            lo, hi, _ = BOUNDS[k]
            st.number_input(
                LABELS[k], min_value=lo, max_value=hi, key=f"nb_{k}", format="%.4f",
                step=1.0 if k.startswith("I") else 0.01, on_change=sync_from, args=(k, "nb"),
            )

cur = np.array([[st.session_state[f"val_{k}"] for k in KEYS]])
p_R, p_Q, svm_class, svm_score = predict_batch(cur)
p = float(p_Q[0])
lr_pred = int(p >= 0.5)
svm_pred = int(svm_class[0])

# ================================================================ header and live result cards
st.title("Three-phase line fault detector")
st.write(
    "Two models read the same six measurements and each decides whether the line has a fault. "
    "Change any value in the sidebar and both answers update."
)


def relay_card(title, subtitle, is_fault, detail_html):
    cls = "fault" if is_fault else "ok"
    word = "Fault" if is_fault else "No fault"
    st.markdown(
        f"""
<div class="relay {cls}">
  <h4>{title}</h4>
  <p class="sub">{subtitle}</p>
  <div class="status"><span class="lamp"></span><span class="word">{word}</span></div>
  {detail_html}
</div>
""",
        unsafe_allow_html=True,
    )


a, b = st.columns(2)
with a:
    relay_card(
        "Logistic regression",
        "Written by hand in NumPy, using the 12 Set Q features",
        bool(lr_pred),
        f"""<div class="small">Probability of fault</div>
<div class="bar"><div style="width:{p*100:.1f}%"></div></div>
<div class="small"><b>{p:.3f}</b> &nbsp;(fault at 0.5 or above)</div>""",
    )
with b:
    relay_card(
        "RBF support vector machine",
        "scikit-learn SVC on the 6 standardised measurements",
        bool(svm_pred),
        f"""<div class="small">Decision score</div>
<div style="font-size:1.25rem;font-weight:600;margin:.25rem 0 .3rem 0">{float(svm_score[0]):+.3f}</div>
<div class="small">Above 0 means fault, below 0 means no fault.</div>""",
    )

st.write("")
if lr_pred == svm_pred:
    st.success("Both models agree.")
else:
    st.warning(
        "The models disagree. A missed fault is the more dangerous error, so treat this "
        "reading as a possible fault until it is checked."
    )

# ================================================================ tabs
tab_sweep, tab_data, tab_w = st.tabs(["Sweep one value", "Browse the dataset", "Model weights"])

# ---------------------------------------------------------------- sweep
with tab_sweep:
    st.write(
        "Pick one measurement and slide it across its range while the other five stay at the values "
        "in the sidebar. This shows where each model changes its mind."
    )
    c_a, c_b = st.columns([1, 2])
    var = c_a.selectbox("Measurement to sweep", KEYS, format_func=lambda k: LABELS[k])
    default_hi = SWEEP_DEFAULT[var]
    lo_b, hi_b, _ = BOUNDS[var]
    r0, r1 = c_b.slider(
        "Range", lo_b, hi_b, (-default_hi, default_hi), key=f"range_{var}",
        step=BOUNDS[var][2] * 10,
    )
    xs = np.linspace(r0, r1, 401)
    base = np.repeat(cur, len(xs), axis=0)
    j = KEYS.index(var)
    base[:, j] = xs
    sR, sQ, _, sScore = predict_batch(base)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    ax1.plot(xs, sR, color="#7A7A7A", lw=2, label="Logistic regression, Set R (raw features)")
    ax1.plot(xs, sQ, color="#2563EB", lw=2, label="Logistic regression, Set Q (with squares)")
    ax1.axhline(0.5, color="k", ls=":", lw=0.8)
    ax1.set_ylabel("P(fault)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(fontsize=8, loc="best")
    ax2.plot(xs, sScore, color="#D1344B", lw=2, label="RBF SVM decision score")
    ax2.fill_between(xs, sScore, 0, where=sScore > 0, color="#D1344B", alpha=0.15)
    ax2.axhline(0, color="k", ls=":", lw=0.8)
    ax2.set_ylabel("SVM score")
    ax2.legend(fontsize=8, loc="best")
    ax2.set_xlabel(LABELS[var])
    for ax in (ax1, ax2):
        ax.axvline(cur[0, j], color="k", ls="--", lw=1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The dashed line marks your current value. The raw-feature logistic regression is a single "
        "straight line in the features, so its probability can only rise toward one side. The Set Q model "
        "and the RBF SVM can raise it on both sides, which is what a large current of either sign needs."
    )

# ---------------------------------------------------------------- dataset browser
with tab_data:
    if data is None:
        st.info("Put `detect_dataset.csv` in a `data/` folder next to `app/` to browse real rows here.")
    else:
        n = len(data)
        st.session_state.setdefault("row", 0)

        def pick_random(kind):
            rng = np.random.default_rng()
            if kind == "any":
                pool = np.arange(n)
            else:
                pool = np.flatnonzero(data["fault"].to_numpy() == (1 if kind == "fault" else 0))
            st.session_state["row"] = int(rng.choice(pool))

        def send_row(i):
            set_inputs({k: float(data.loc[i, k]) for k in KEYS})

        st.write("Step through real rows, see the true label, and check whether each model got it right.")
        g1, g2, g3 = st.columns(3)
        g1.button("Random row", key="rand_any", on_click=pick_random, args=("any",))
        g2.button("Random fault row", key="rand_fault", on_click=pick_random, args=("fault",))
        g3.button("Random no-fault row", key="rand_nofault", on_click=pick_random, args=("nofault",))
        i = st.slider("Row number", 0, n - 1, key="row")

        row = data.loc[i]
        truth = int(row["fault"])
        vals = np.array([[row[k] for k in KEYS]], dtype=float)
        rR, rQ, rClass, rScore = predict_batch(vals)
        lr_row, svm_row = int(rQ[0] >= 0.5), int(rClass[0])

        st.dataframe(pd.DataFrame([{k: round(float(row[k]), 4) for k in KEYS}]), hide_index=True)
        st.write(
            f"True label: **{'fault' if truth else 'no fault'}**. "
            f"This row belongs to the **{row['split']}** split"
            + (" (the models never saw it while training)." if row["split"] == "test" else " (the models used it for training or tuning).")
        )
        d1, d2 = st.columns(2)
        for col, name, pred in ((d1, "Logistic regression (Set Q)", lr_row), (d2, "RBF SVM", svm_row)):
            msg = f"{name}: {'fault' if pred else 'no fault'}"
            if pred == truth:
                col.success(msg + ", correct")
            else:
                col.error(msg + (", missed fault" if truth == 1 else ", false alarm"))
        st.button("Send this row to the sidebar sliders", key="send_row", on_click=send_row, args=(i,))

# ---------------------------------------------------------------- weights
with tab_w:
    names = ["intercept"] + KEYS + [f"{k}²" for k in KEYS]
    st.write(
        "Learned weights of the Set Q logistic regression, all features standardised. A positive weight "
        "pushes toward fault, a negative weight pushes toward no fault."
    )
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = ["#D1344B" if w > 0 else "#2E9E6B" for w in theta_Q]
    ax.barh(names[::-1], theta_Q[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("weight")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        pd.DataFrame({"feature": names, "weight": np.round(theta_Q, 4)}), hide_index=True
    )

with st.expander("How the numbers are prepared"):
    st.markdown(
        "- Every value is standardised with the **training** mean and standard deviation saved by the notebook.\n"
        "- For Set Q the six squares are computed from the raw values first, then all 12 columns are "
        "standardised, then a constant 1 is added for the intercept.\n"
        "- The RBF SVM receives only the six standardised measurements."
    )

st.caption("Test-set accuracy from the assignment run: logistic regression (Set Q) 97.88%, RBF SVM 99.58%.")